#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

int enzyme_dup;
template <typename return_type, typename... T>
return_type __enzyme_fp_optimize(void *, T...);

// simple vector / matrix functions

struct vec3 {
  double data[3];
  double &operator[](int i) { return data[i]; }
  const double &operator[](int i) const { return data[i]; }
};

struct mat3 {
  vec3 data[3];
  vec3 &operator[](int i) { return data[i]; }
  const vec3 &operator[](int i) const { return data[i]; }
};

double dot(const vec3 &u, const vec3 &v) {
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

double norm(const vec3 &v) {
  return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

vec3 normalize(const vec3 &v) {
  double inv_normv = 1.0 / norm(v);
  return vec3{v[0] * inv_normv, v[1] * inv_normv, v[2] * inv_normv};
}

vec3 cross(const vec3 &u, const vec3 &v) {
  return vec3{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
              u[0] * v[1] - u[1] * v[0]};
}

vec3 operator-(const vec3 &u, const vec3 &v) {
  return vec3{u[0] - v[0], u[1] - v[1], u[2] - v[2]};
}

vec3 operator+(const vec3 &u, const vec3 &v) {
  return vec3{u[0] + v[0], u[1] + v[1], u[2] + v[2]};
}

vec3 operator*(const vec3 &v, double scale) {
  return vec3{v[0] * scale, v[1] * scale, v[2] * scale};
}

vec3 dot(const mat3 &A, const vec3 &v) {
  return vec3{dot(A[0], v), dot(A[1], v), dot(A[2], v)};
}

double tr(const mat3 &A) { return A[0][0] + A[1][1] + A[2][2]; }

double second_invariant(const mat3 &A) {
  return A[0][1] * A[1][0] - A[0][0] * A[1][1] + A[0][2] * A[2][0] +
         A[1][2] * A[2][1] - A[0][0] * A[2][2] - A[1][1] * A[2][2];
}

double det(const mat3 &A) {
  return A[0][0] * A[1][1] * A[2][2] + A[0][1] * A[1][2] * A[2][0] +
         A[0][2] * A[1][0] * A[2][1] - A[0][0] * A[1][2] * A[2][1] -
         A[0][1] * A[1][0] * A[2][2] - A[0][2] * A[1][1] * A[2][0];
}

double signum(double x) {
  if (x < 0) {
    return -1.0;
  }
  if (x > 0) {
    return +1.0;
  }
  return 0.0; // if x == 0.0
}

std::ostream &operator<<(std::ostream &out, vec3 v) {
  out << '{' << v[0];
  for (int i = 1; i < 3; i++) {
    out << ", " << v[i];
  }
  out << '}';
  return out;
}

std::ostream &operator<<(std::ostream &out, const mat3 &A) {
  out << '{';
  for (int i = 0; i < 3; i++) {
    out << A[i];
    if (i != 2)
      out << ',';
  }
  out << '}';

  return out;
}

////////////////////////////////////////////////////////////////////////////////

struct Eigensystem {
  vec3 lambda;
  mat3 X;
};

// eigendecomposition for **symmetric** A
//
// based on "A robust algorithm for finding the eigenvalues and
// eigenvectors of 3x3 symmetric matrices", by Scherzinger & Dohrmann
void eig(const mat3 &A, Eigensystem &res) {

  vec3 eigenvalues = {0.0, 0.0, 0.0};
  mat3 eigenvectors = {{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};

  double trA_3 = tr(A) / 3.0;
  mat3 A_dev = A;
  for (int i = 0; i < 3; i++) {
    A_dev[i][i] -= trA_3;
  }

  double J2 = second_invariant(A_dev);
  double J3 = det(A_dev);

  if (J2 > 0.0) {

    // angle used to find eigenvalues
    double tmp = (0.5 * J3) * pow(3.0 / J2, 1.5);
    double alpha = acos(fmin(fmax(tmp, -1.0), 1.0)) / 3.0;

    // consider the most distinct eigenvalue first
    if (6.0 * alpha < M_PI) {
      eigenvalues[0] = 2 * sqrt(J2 / 3.0) * cos(alpha);
    } else {
      eigenvalues[0] = 2 * sqrt(J2 / 3.0) * cos(alpha + 2.0 * M_PI / 3.0);
    }

    // find the eigenvector for that eigenvalue
    vec3 r[3];

    int imax = -1;
    double norm_max = -1.0;

    for (int i = 0; i < 3; i++) {

      for (int j = 0; j < 3; j++) {
        r[i][j] = A_dev[j][i] - (i == j) * eigenvalues[0];
      }

      double norm_r = norm(r[i]);
      if (norm_max < norm_r) {
        imax = i;
        norm_max = norm_r;
      }
    }

    vec3 s0, s1, t1, t2, v0, v1, v2, w;

    s0 = normalize(r[imax]);
    t1 = r[(imax + 1) % 3] - s0 * dot(r[(imax + 1) % 3], s0);
    t2 = r[(imax + 2) % 3] - s0 * dot(r[(imax + 2) % 3], s0);
    s1 = normalize((norm(t1) > norm(t2)) ? t1 : t2);

    // record the first eigenvector
    v0 = cross(s0, s1);
    eigenvectors[0] = v0;

    // get the other two eigenvalues by solving the
    // remaining quadratic characteristic polynomial
    auto A_dev_s0 = dot(A_dev, s0);
    auto A_dev_s1 = dot(A_dev, s1);

    double A11 = dot(s0, A_dev_s0);
    double A12 = dot(s0, A_dev_s1);
    double A21 = dot(s1, A_dev_s0);
    double A22 = dot(s1, A_dev_s1);

    double delta = 0.5 * signum(A11 - A22) *
                   sqrt((A11 - A22) * (A11 - A22) + 4 * A12 * A21);

    eigenvalues[1] = 0.5 * (A11 + A22) - delta;
    eigenvalues[2] = 0.5 * (A11 + A22) + delta;

    // if the remaining eigenvalues are exactly the same
    // then just use the basis for the orthogonal complement
    // found earlier
    if (fabs(delta) <= 1.0e-15) {

      eigenvectors[1] = s0;
      eigenvectors[2] = s1;

      // otherwise compute the remaining eigenvectors
    } else {

      t1 = A_dev_s0 - s0 * eigenvalues[1];
      t2 = A_dev_s1 - s1 * eigenvalues[1];

      w = normalize((norm(t1) > norm(t2)) ? t1 : t2);

      v1 = normalize(cross(w, v0));
      eigenvectors[1] = v1;

      // define the last eigenvector as
      // the direction perpendicular to the
      // first two directions
      v2 = normalize(cross(v0, v1));
      eigenvectors[2] = v2;
    }
  }

  // eta are actually eigenvalues of A_dev, so
  // shift them to get eigenvalues of A
  eigenvalues = eigenvalues + vec3{1, 1, 1} * trA_3;

  res.lambda = eigenvalues;
  res.X = eigenvectors;
}

int main(int argc, char *argv[]) {
  int N = 1e5;
  std::string output_path = "";
  std::string case_type = "equal";
  bool save_output = false;
  bool profiling_mode = false;

  // Parse command-line arguments
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
    } else if (strcmp(argv[i], "--N") == 0) {
      if (i + 1 < argc) {
        N = std::stoi(argv[++i]);
      } else {
        std::cerr << "Error: --N option requires a value" << std::endl;
        return 1;
      }
    } else if (strcmp(argv[i], "--prof") == 0) {
      profiling_mode = true;
      N = 1e1;
    } else if (strcmp(argv[i], "--case") == 0) {
      if (i + 1 < argc) {
        case_type = argv[++i];
        if (case_type != "biased" && case_type != "equal") {
          std::cerr << "Error: --case must be 'biased' or 'equal'" << std::endl;
          return 1;
        }
      } else {
        std::cerr << "Error: --case option requires a value" << std::endl;
        return 1;
      }
    }
  }

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
  }

  std::mt19937 rd;
  rd.seed(profiling_mode ? 123 : 42); // Different seed for profiling
  std::uniform_real_distribution<double> dist(-0.1, 0.1);

  // Timing
  auto start_time = std::chrono::high_resolution_clock::now();

  for (int iter = 0; iter < N; ++iter) {
    mat3 S;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        S[i][j] = dist(rd);

    mat3 A;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        A[i][j] = S[i][j] + S[j][i];
        if (i == j)
          A[i][j] += 1.0;
      }

    Eigensystem res;
    mat3 A_grad;
    Eigensystem res_grad;

    if (case_type == "biased") {
      // Case: 1, 1e-5, 1e-5, 0, 0, 0
      res_grad.lambda[0] = 1.0;
      for (int i = 1; i < 3; ++i)
        res_grad.lambda[i] = 1e-5;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          res_grad.X[i][j] = 0.0;
    } else {
      // Case: 1, 1, 1, 1, 1, 1
      res_grad.lambda[0] = 1.0;
      for (int i = 1; i < 3; ++i)
        res_grad.lambda[i] = 1.0;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          res_grad.X[i][j] = 1.0;
    }

    __enzyme_fp_optimize<void>((void *)eig, enzyme_dup, &A, &A_grad, enzyme_dup,
                               &res, &res_grad);

    if (save_output) {
      outfile << "### Iteration: " << iter << "\n\n";
      // Write eigenvalues
      outfile << "Eigenvalues:\n[";
      for (int i = 0; i < 3; ++i) {
        outfile << res.lambda[i];
        if (i < 2)
          outfile << ", ";
      }
      outfile << "]\n\n";

      // Write eigenvectors
      outfile << "Eigenvectors:\n";
      for (int i = 0; i < 3; ++i) {
        outfile << "[";
        for (int j = 0; j < 3; ++j) {
          outfile << res.X[i][j];
          if (j < 2)
            outfile << ", ";
        }
        outfile << "]\n";
      }
      outfile << "\n";
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();

  // Compute runtime
  std::chrono::duration<double> elapsed = end_time - start_time;

  // Print runtime to stdout
  std::cout << "Elapsed time = " << elapsed.count() << " (s)\n";

  // Close output file if it was opened
  if (save_output) {
    outfile.close();
    std::cout << "Results saved to: " << output_path << std::endl;
  }

  return 0;
}
