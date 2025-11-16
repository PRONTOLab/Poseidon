#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>

int enzyme_dup;
template <typename return_type, typename... T>
return_type __enzyme_fp_optimize(void *, T...);

constexpr size_t invdim = 4;
constexpr size_t emapdim = 3;
constexpr size_t nwvec = 3;
constexpr size_t qdim = 4;

constexpr double idp_tiny_sqrt = 1.0e-90;
constexpr double zero = 0.0;
constexpr double one = 1.0;
constexpr double onehalf = 0.5;
constexpr double oneqrtr = 0.25;

#define ECMECH_NM_INDX(p, q, qDim) ((p) * (qDim) + (q))

// Simple scalar product scaling function
template <int n>
inline void vecsVxa(double *const v, double x, const double *const a) {
  for (int i = 0; i < n; ++i) {
    v[i] = x * a[i];
  }
}

// The L2 norm of a vector
template <int n> inline double vecNorm(const double *const v) {
  double retval = 0.0;
  for (int i = 0; i < n; ++i) {
    retval += v[i] * v[i];
  }

  retval = sqrt(retval);
  return retval;
}

inline void inv_to_quat(double *const quat, const double *const inv) {
  double a = inv[0] * 0.5;
  quat[0] = cos(a);
  a = sin(a);
  vecsVxa<nwvec>(&(quat[1]), a, &(inv[1]));
}

__attribute__((noinline)) void emap_to_quat(double *const quat,
                                            const double *const emap) {
  double inv[invdim] = {0.0, 1.0, 0.0, 0.0};
  inv[0] = vecNorm<emapdim>(emap);
  if (inv[0] > idp_tiny_sqrt) {
    double invInv = 1.0 / inv[0];
    vecsVxa<emapdim>(&(inv[1]), invInv, emap);
  } // else, emap is effectively zero, so axis does not matter
  inv_to_quat(quat, inv);
}

/**
    ! derivative of quaternion parameters with respect to exponential map
    ! parameters
    ! This is the derivative of emap_to_quat function with a transpose applied
*/
__attribute__((noinline)) void
dquat_demap_T(double *const dqdeT_raw, // (emapdim, qdim)
              const double *const emap // (emapdim)
) {
  const double theta_sm_a = 1e-9;
  const double oo48 = 1.0 / 48.0;

  double theta = vecNorm<emapdim>(emap);

  double theta_inv, sthhbyth, halfsthh, na, nb, nc;
  // When dealing with exponential maps that have very small rotations
  // along an axis the derivative term can become difficult to
  // calculate with the analytic solution
  //
  if (fabs(theta) < theta_sm_a) {
    sthhbyth = onehalf -
               theta * theta * oo48; // truncated Taylor series; probably safe
                                     // to just use onehalf and be done with it
    halfsthh = theta * oneqrtr;      // truncated Taylor series
    if (fabs(theta) < idp_tiny_sqrt) {
      // n is arbitrary, as theta is effectively zero
      na = one;
      nb = zero;
      nc = zero;
    } else {
      theta_inv = one / theta;
      na = emap[0] * theta_inv;
      nb = emap[1] * theta_inv;
      nc = emap[2] * theta_inv;
    }
  } else {
    halfsthh = sin(theta * onehalf);
    sthhbyth = halfsthh / theta;
    halfsthh = halfsthh * onehalf;
    theta_inv = one / theta;
    na = emap[0] * theta_inv;
    nb = emap[1] * theta_inv;
    nc = emap[2] * theta_inv;
  }
  //
  double halfcthh = cos(theta * onehalf) * onehalf;
  //
  // now have: halfsthh, sthhbyth, halfcthh, theta, na, nb, nc
  // If we made use of something like RAJA or Kokkos views or mdspans then we
  // could make some of the indexing here nicer but don't worry about that now
  // RAJA::View<double, RAJA::Layout<2> > dqdeT(dqdeT_raw, emapdim, qdim);

  dqdeT_raw[ECMECH_NM_INDX(0, 0, qdim)] = -halfsthh * na;
  dqdeT_raw[ECMECH_NM_INDX(1, 0, qdim)] = -halfsthh * nb;
  dqdeT_raw[ECMECH_NM_INDX(2, 0, qdim)] = -halfsthh * nc;

  double temp = na * na;
  dqdeT_raw[ECMECH_NM_INDX(0, 1, qdim)] =
      halfcthh * temp + sthhbyth * (one - temp);
  //
  temp = nb * nb;
  dqdeT_raw[ECMECH_NM_INDX(1, 2, qdim)] =
      halfcthh * temp + sthhbyth * (one - temp);
  //
  temp = nc * nc;
  dqdeT_raw[ECMECH_NM_INDX(2, 3, qdim)] =
      halfcthh * temp + sthhbyth * (one - temp);

  temp = halfcthh - sthhbyth;
  //
  double tempb = temp * na * nb;
  dqdeT_raw[ECMECH_NM_INDX(1, 1, qdim)] = tempb;
  dqdeT_raw[ECMECH_NM_INDX(0, 2, qdim)] = tempb;
  //
  tempb = temp * na * nc;
  dqdeT_raw[ECMECH_NM_INDX(2, 1, qdim)] = tempb;
  dqdeT_raw[ECMECH_NM_INDX(0, 3, qdim)] = tempb;
  //
  tempb = temp * nb * nc;
  dqdeT_raw[ECMECH_NM_INDX(2, 2, qdim)] = tempb;
  dqdeT_raw[ECMECH_NM_INDX(1, 3, qdim)] = tempb;
}

int main(int argc, char *argv[]) {
  int num_tests = 1e6;
  std::string output_path = "";
  bool save_output = false;
  int seed = 42;
  bool profiling_mode = false;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--output-path") == 0) {
      if (i + 1 < argc) {
        output_path = argv[++i];
        save_output = true;
      } else {
        std::cerr << "Error: --output-path option requires a value"
                  << std::endl;
        return 1;
      }
    } else if (strcmp(argv[i], "--profiling") == 0) {
      profiling_mode = true;
    } else if (strcmp(argv[i], "--num-tests") == 0) {
      if (i + 1 < argc) {
        num_tests = std::atoi(argv[++i]);
      } else {
        std::cerr << "Error: --num-tests option requires a value" << std::endl;
        return 1;
      }
    } else if (strcmp(argv[i], "--seed") == 0) {
      if (i + 1 < argc) {
        seed = std::atoi(argv[++i]);
      } else {
        std::cerr << "Error: --seed option requires a value" << std::endl;
        return 1;
      }
    } else {
      std::cerr << "Unknown option: " << argv[i] << std::endl;
      std::cerr << "Usage: " << argv[0]
                << " [--output-path <path>] [--profiling] [--num-tests <n>] "
                   "[--seed <s>]"
                << std::endl;
      return 1;
    }
  }

  if (profiling_mode) {
    if (argc < 4 || strstr(argv[0], "--num-tests") == nullptr) {
      num_tests = 1000;
    }
    if (argc < 4 || strstr(argv[0], "--seed") == nullptr) {
      seed = 123;
    }
    std::cout << "Profiling mode: using " << num_tests << " tests with seed "
              << seed << std::endl;
  }

  std::ostream *out_stream = &std::cout;
  std::ofstream outfile;
  if (save_output) {
    outfile.open(output_path);
    if (!outfile) {
      std::cerr << "Error: Could not open output file: " << output_path
                << std::endl;
      return 1;
    }
    outfile << std::setprecision(std::numeric_limits<double>::digits10 + 1)
            << std::scientific;
    out_stream = &outfile;
  }

  std::mt19937 gen;
  gen.seed(seed);

  std::uniform_real_distribution<> dis_case1(-1e-9,
                                             1e-9); // 1e-90 <= theta < 1e-9
  std::uniform_real_distribution<> dis_case2(-1e-1, 1e-1); // theta >= 1e-9

  const int num_cases = 2;

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int test = 0; test < num_tests; ++test) {
    double emap[emapdim] = {};

    int current_case = test % num_cases;

    switch (current_case) {
    case 0:
      for (size_t i = 0; i < emapdim; i++) {
        emap[i] = dis_case1(gen);
      }
      break;
    case 1:
      for (size_t i = 0; i < emapdim; i++) {
        emap[i] = dis_case2(gen);
      }
      break;
    default:
      std::cerr << "Invalid case encountered!" << std::endl;
      return 1;
    }

    double dquat_dexpmap_t[emapdim * qdim] = {};
    double dquat_dexpmap_t_grad[emapdim * qdim];
    std::fill(dquat_dexpmap_t_grad, dquat_dexpmap_t_grad + emapdim * qdim, 1.0);
    double emap_grad[emapdim];
    std::fill(emap_grad, emap_grad + emapdim, 0.0);
    __enzyme_fp_optimize<void>((void *)dquat_demap_T, enzyme_dup,
                               dquat_dexpmap_t, dquat_dexpmap_t_grad,
                               enzyme_dup, emap, emap_grad);

    if (save_output) {
      *out_stream << "Test " << test + 1 << ":\n";

      *out_stream << "exponential map value:\n";
      for (size_t ie = 0; ie < emapdim; ie++) {
        *out_stream << emap[ie] << " ";
      }
      *out_stream << "\n";

      //   *out_stream << "unit quaternion value:\n";
      //   for (size_t iq = 0; iq < qdim; iq++) {
      //     *out_stream << quat[iq] << " ";
      //   }
      //   *out_stream << "\n";

      *out_stream << "dquat_dexpmap ^ T value:\n";
      for (size_t ie = 0; ie < emapdim; ie++) {
        for (size_t iq = 0; iq < qdim; iq++) {
          *out_stream << dquat_dexpmap_t[ECMECH_NM_INDX(ie, iq, qdim)] << " ";
        }
        *out_stream << "\n";
      }
      *out_stream << "----------------------------------------\n";
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end_time - start_time;

  std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1)
            << std::scientific << "Elapsed time = " << elapsed.count()
            << " (s)\n";

  if (save_output) {
    outfile.close();
    std::cout << "Results saved to: " << output_path << std::endl;
  }

  return 0;
}
