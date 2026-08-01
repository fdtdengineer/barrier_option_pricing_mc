#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace barrier {

inline double normal_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

inline bool valid_inputs(double spot, double strike, double barrier, double maturity,
                         double volatility, std::uint64_t n_paths = 1, int n_steps = 1) {
    return spot > 0.0 && strike > 0.0 && barrier > 0.0 && maturity > 0.0 &&
           volatility > 0.0 && n_paths > 0 && n_steps > 0;
}

// Acklam's inverse-normal approximation. The input is clamped away from {0,1}.
inline double inverse_normal_cdf(double p) {
    constexpr double a1 = -3.969683028665376e+01;
    constexpr double a2 =  2.209460984245205e+02;
    constexpr double a3 = -2.759285104469687e+02;
    constexpr double a4 =  1.383577518672690e+02;
    constexpr double a5 = -3.066479806614716e+01;
    constexpr double a6 =  2.506628277459239e+00;
    constexpr double b1 = -5.447609879822406e+01;
    constexpr double b2 =  1.615858368580409e+02;
    constexpr double b3 = -1.556989798598866e+02;
    constexpr double b4 =  6.680131188771972e+01;
    constexpr double b5 = -1.328068155288572e+01;
    constexpr double c1 = -7.784894002430293e-03;
    constexpr double c2 = -3.223964580411365e-01;
    constexpr double c3 = -2.400758277161838e+00;
    constexpr double c4 = -2.549732539343734e+00;
    constexpr double c5 =  4.374664141464968e+00;
    constexpr double c6 =  2.938163982698783e+00;
    constexpr double d1 =  7.784695709041462e-03;
    constexpr double d2 =  3.224671290700398e-01;
    constexpr double d3 =  2.445134137142996e+00;
    constexpr double d4 =  3.754408661907416e+00;
    constexpr double p_low = 0.02425;
    constexpr double p_high = 1.0 - p_low;

    p = std::min(std::max(p, std::numeric_limits<double>::epsilon()),
                 1.0 - std::numeric_limits<double>::epsilon());
    if (p < p_low) {
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
               ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
    }
    if (p <= p_high) {
        const double q = p - 0.5;
        const double r = q * q;
        return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q /
               (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1.0);
    }
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
            ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
}

inline double analytic_up_and_out_call(double spot, double strike, double barrier,
                                       double maturity, double rate,
                                       double dividend_yield, double volatility) {
    if (!valid_inputs(spot, strike, barrier, maturity, volatility)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (spot >= barrier || strike >= barrier) {
        return 0.0;
    }

    const double stddev = volatility * std::sqrt(maturity);
    const double mu = (rate - dividend_yield) / (volatility * volatility) - 0.5;
    const double mu_sigma = (1.0 + mu) * stddev;
    const double risk_discount = std::exp(-rate * maturity);
    const double dividend_discount = std::exp(-dividend_yield * maturity);

    const auto vanilla_piece = [&](double reference) {
        const double x = std::log(spot / reference) / stddev + mu_sigma;
        return spot * dividend_discount * normal_cdf(x)
             - strike * risk_discount * normal_cdf(x - stddev);
    };

    const double A = vanilla_piece(strike);
    const double B = vanilla_piece(barrier);
    const double hs = barrier / spot;
    const double pow0 = std::pow(hs, 2.0 * mu);
    const double pow1 = pow0 * hs * hs;

    const auto reflected_piece = [&](double reflected_reference) {
        const double y = std::log(reflected_reference) / stddev + mu_sigma;
        // eta=-1, phi=+1 for an up barrier call.
        return spot * dividend_discount * pow1 * normal_cdf(-y)
             - strike * risk_discount * pow0 * normal_cdf(-(y - stddev));
    };

    const double C = reflected_piece(barrier * hs / strike);
    const double D = reflected_piece(barrier / spot);
    return std::max(0.0, A - B + C - D);
}

inline double bridge_survival_factor(double log_s0, double log_s1, double log_barrier,
                                     double variance_step) {
    if (log_s0 >= log_barrier || log_s1 >= log_barrier) {
        return 0.0;
    }
    const double exponent = -2.0 * (log_barrier - log_s0) *
                            (log_barrier - log_s1) / variance_step;
    const double hit_probability = std::exp(exponent);
    return std::max(0.0, 1.0 - hit_probability);
}

}  // namespace barrier
