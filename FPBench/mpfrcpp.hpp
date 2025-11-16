// Adapted from https://github.com/jhueckelheim/force/blob/master/include/mpfrcpp_tpl.h
#ifndef mpfrcpp_h
#define mpfrcpp_h

#include <gmp.h>
#include <iostream>
#include <mpfr.h>
#include <stdio.h>
#include <stdlib.h>

template <unsigned int MPFRPREC> class mpfrcpp {
public:
  mpfr_t value;

  mpfrcpp() { mpfr_init2(value, MPFRPREC); }
  mpfrcpp(const float v) {
    mpfr_init2(value, MPFRPREC);
    mpfr_set_flt(value, v, MPFR_RNDN);
  }
  mpfrcpp(const double v) {
    mpfr_init2(value, MPFRPREC);
    mpfr_set_d(value, v, MPFR_RNDN);
  }
  mpfrcpp(const long double v) {
    mpfr_init2(value, MPFRPREC);
    mpfr_set_ld(value, v, MPFR_RNDN);
  }
  mpfrcpp(const mpfrcpp &v) {
    mpfr_init2(value, MPFRPREC);
    mpfr_set(value, v.value, MPFR_RNDN);
  }
  mpfrcpp(const mpfr_t &v) {
    mpfr_init2(value, MPFRPREC);
    mpfr_set(value, v, MPFR_RNDN);
  }
  mpfrcpp(const char *v) {
    mpfr_init2(value, MPFRPREC);
    mpfr_set_str(value, v, 10, MPFR_RNDN);
  }
  ~mpfrcpp() { mpfr_clear(value); }

  mpfrcpp &operator=(const mpfrcpp &g1) {
    mpfr_set(value, g1.value, MPFR_RNDN);
    return *this;
  }
  mpfrcpp &operator=(const double &g1) {
    mpfr_set_d(value, g1, MPFR_RNDN);
    return *this;
  }

  mpfrcpp &operator+=(const double &g1) {
    mpfr_add_d(value, value, g1, MPFR_RNDN);
    return *this;
  }
  mpfrcpp &operator-=(const double &g1) {
    mpfr_sub_d(value, value, g1, MPFR_RNDN);
    return *this;
  }
  mpfrcpp &operator*=(const double &g1) {
    mpfr_mul_d(value, value, g1, MPFR_RNDN);
    return *this;
  }
  mpfrcpp &operator/=(const double &g1) {
    mpfr_div_d(value, value, g1, MPFR_RNDN);
    return *this;
  }

  operator double() const { return mpfr_get_d(value, MPFR_RNDN); }
};

template <unsigned int MPFRPREC>
mpfrcpp<MPFRPREC> operator+(const mpfrcpp<MPFRPREC> &g1, const double &g2) {
  mpfrcpp<MPFRPREC> res;
  mpfr_add_d(res.value, g1.value, g2, MPFR_RNDN);
  return res;
}

template <unsigned int MPFRPREC>
mpfrcpp<MPFRPREC> operator-(const mpfrcpp<MPFRPREC> &g1, const double &g2) {
  mpfrcpp<MPFRPREC> res;
  mpfr_sub_d(res.value, g1.value, g2, MPFR_RNDN);
  return res;
}

template <unsigned int MPFRPREC>
mpfrcpp<MPFRPREC> operator*(const mpfrcpp<MPFRPREC> &g1, const double &g2) {
  mpfrcpp<MPFRPREC> res;
  mpfr_mul_d(res.value, g1.value, g2, MPFR_RNDN);
  return res;
}

template <unsigned int MPFRPREC>
mpfrcpp<MPFRPREC> operator/(const mpfrcpp<MPFRPREC> &g1, const double &g2) {
  mpfrcpp<MPFRPREC> res;
  mpfr_div_d(res.value, g1.value, g2, MPFR_RNDN);
  return res;
}

template <unsigned int MPFRPREC>
mpfrcpp<MPFRPREC> operator+(const double &g1, const mpfrcpp<MPFRPREC> &g2) {
  mpfrcpp<MPFRPREC> res;
  mpfr_add_d(res.value, g2.value, g1, MPFR_RNDN);
  return res;
}

template <unsigned int MPFRPREC>
mpfrcpp<MPFRPREC> operator-(const double &g1, const mpfrcpp<MPFRPREC> &g2) {
  mpfrcpp<MPFRPREC> res;
  mpfr_d_sub(res.value, g1, g2.value, MPFR_RNDN);
  return res;
}

template <unsigned int MPFRPREC>
mpfrcpp<MPFRPREC> operator*(const double &g1, const mpfrcpp<MPFRPREC> &g2) {
  mpfrcpp<MPFRPREC> res;
  mpfr_mul_d(res.value, g2.value, g1, MPFR_RNDN);
  return res;
}

template <unsigned int MPFRPREC>
mpfrcpp<MPFRPREC> operator/(const double &g1, const mpfrcpp<MPFRPREC> &g2) {
  mpfrcpp<MPFRPREC> res;
  mpfr_d_div(res.value, g1, g2.value, MPFR_RNDN);
  return res;
}

template <unsigned int MPFRPREC>
mpfrcpp<MPFRPREC> operator+(const mpfrcpp<MPFRPREC> &g1,
                            const mpfrcpp<MPFRPREC> &g2) {
  mpfrcpp<MPFRPREC> res;
  mpfr_add(res.value, g1.value, g2.value, MPFR_RNDN);
  return res;
}

template <unsigned int MPFRPREC>
mpfrcpp<MPFRPREC> operator-(const mpfrcpp<MPFRPREC> &g1,
                            const mpfrcpp<MPFRPREC> &g2) {
  mpfrcpp<MPFRPREC> res;
  mpfr_sub(res.value, g1.value, g2.value, MPFR_RNDN);
  return res;
}

template <unsigned int MPFRPREC>
mpfrcpp<MPFRPREC> operator*(const mpfrcpp<MPFRPREC> &g1,
                            const mpfrcpp<MPFRPREC> &g2) {
  mpfrcpp<MPFRPREC> res;
  mpfr_mul(res.value, g1.value, g2.value, MPFR_RNDN);
  return res;
}

template <unsigned int MPFRPREC>
mpfrcpp<MPFRPREC> operator/(const mpfrcpp<MPFRPREC> &g1,
                            const mpfrcpp<MPFRPREC> &g2) {
  mpfrcpp<MPFRPREC> res;
  mpfr_div(res.value, g1.value, g2.value, MPFR_RNDN);
  return res;
}

template <unsigned int MPFRPREC>
bool operator>=(const mpfrcpp<MPFRPREC> &g1, const double &g2) {
  return mpfr_cmp_d(g1.value, g2) >= 0;
}

template <unsigned int MPFRPREC>
bool operator>(const mpfrcpp<MPFRPREC> &g1, const double &g2) {
  return mpfr_cmp_d(g1.value, g2) > 0;
}

template <unsigned int MPFRPREC>
bool operator<=(const mpfrcpp<MPFRPREC> &g1, const double &g2) {
  return mpfr_cmp_d(g1.value, g2) <= 0;
}

template <unsigned int MPFRPREC>
bool operator<(const mpfrcpp<MPFRPREC> &g1, const double &g2) {
  return mpfr_cmp_d(g1.value, g2) < 0;
}

template <unsigned int MPFRPREC>
bool operator>=(const double &g1, const mpfrcpp<MPFRPREC> &g2) {
  return mpfr_cmp_d(g2.value, g1) <= 0;
}

template <unsigned int MPFRPREC>
bool operator>(const double &g1, const mpfrcpp<MPFRPREC> &g2) {
  return mpfr_cmp_d(g2.value, g1) < 0;
}

template <unsigned int MPFRPREC>
bool operator<=(const double &g1, const mpfrcpp<MPFRPREC> &g2) {
  return mpfr_cmp_d(g2.value, g1) >= 0;
}

template <unsigned int MPFRPREC>
bool operator<(const double &g1, const mpfrcpp<MPFRPREC> &g2) {
  return mpfr_cmp_d(g2.value, g1) > 0;
}

template <unsigned int MPFRPREC>
int operator>=(const mpfrcpp<MPFRPREC> &g1, const mpfrcpp<MPFRPREC> &g2) {
  return mpfr_greaterequal_p(g1.value, g2.value);
}

template <unsigned int MPFRPREC>
int operator>(const mpfrcpp<MPFRPREC> &g1, const mpfrcpp<MPFRPREC> &g2) {
  return mpfr_greater_p(g1.value, g2.value);
}

template <unsigned int MPFRPREC>
int operator<(const mpfrcpp<MPFRPREC> &g1, const mpfrcpp<MPFRPREC> &g2) {
  return mpfr_greater_p(g2.value, g1.value);
}

template <unsigned int MPFRPREC>
mpfrcpp<MPFRPREC> fabs(const mpfrcpp<MPFRPREC> &g1) {
  mpfrcpp<MPFRPREC> res;
  mpfr_abs(res.value, g1.value, MPFR_RNDN);
  return res;
}

template <unsigned int MPFRPREC>
mpfrcpp<MPFRPREC> pow(const mpfrcpp<MPFRPREC> &g1, double expd) {
  mpfrcpp<MPFRPREC> res;
  mpfr_t exp_mpfr;
  mpfr_init2(exp_mpfr, MPFRPREC);
  mpfr_set_d(exp_mpfr, expd, MPFR_RNDN);
  mpfr_pow(res.value, g1.value, exp_mpfr, MPFR_RNDN);
  mpfr_clear(exp_mpfr);
  return res;
}

template <unsigned int MPFRPREC>
mpfrcpp<MPFRPREC> sqrt(const mpfrcpp<MPFRPREC> &g1) {
  mpfrcpp<MPFRPREC> res;
  mpfr_sqrt(res.value, g1.value, MPFR_RNDN);
  return res;
}

template <unsigned int MPFRPREC>
std::ostream &operator<<(std::ostream &ost, const mpfrcpp<MPFRPREC> &ad) {
  char *abc = NULL;
  mpfr_exp_t i;
  if (ad >= mpfrcpp<MPFRPREC>(0.0)) {
    abc = mpfr_get_str(NULL, &i, 10, 0, ad.value, MPFR_RNDN);
    ost << "0." << abc << "e" << i;
  } else {
    abc = mpfr_get_str(NULL, &i, 10, 0, (-ad).value, MPFR_RNDN);
    ost << "-0." << abc << "e" << i;
  }
  mpfr_free_str(abc);
  return ost;
}

template <unsigned int fromprec, unsigned int toprec>
mpfrcpp<toprec> convert(const mpfrcpp<fromprec> &from) {
  mpfrcpp<toprec> to;
  mpfr_set(to.value, from.value, MPFR_RNDN);
  return to;
}

#endif
