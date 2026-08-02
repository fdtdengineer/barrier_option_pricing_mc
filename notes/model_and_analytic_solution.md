# Black–Scholes model and analytic up-and-out call solution

## Scope

This note derives the closed-form price implemented by
`analytic_up_and_out_call` in `src/barrier_common.hpp`. The contract is a
continuously monitored European up-and-out call with a constant upper barrier
and zero rebate.

The derivation assumes

- a Black–Scholes market with constant interest rate `r`, dividend yield `q`,
  and volatility `sigma`;
- an initial spot `S0 > 0`;
- strike `K > 0`, upper barrier `H > 0`, and maturity `T > 0`;
- continuous barrier monitoring;
- immediate knockout when the spot reaches or exceeds `H`;
- no payment after knockout.

The nontrivial case is

$$
0 < S_0 < H, \qquad 0 < K < H.
$$

If $S_0 \ge H$, the option is already knocked out. If $K \ge H$, every path
that survives to maturity has $S_T < H \le K$, so its call payoff is zero.

## 1. Risk-neutral model and contract value

Under the risk-neutral measure $\mathbb{Q}$, the underlying follows geometric
Brownian motion,

$$
\frac{dS_t}{S_t}=(r-q)\,dt+\sigma\,dW_t.
$$

Its explicit solution is

$$
S_t=S_0\exp\left[
\left(r-q-\frac{1}{2}\sigma^2\right)t+\sigma W_t
\right].
$$

Define the first hitting time of the upper barrier by

$$
\tau_H=\inf\{t\ge 0:S_t\ge H\}.
$$

The time-zero value of the zero-rebate up-and-out call is therefore

$$
C_{\mathrm{UO}}
=e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\left[
(S_T-K)^+\mathbf{1}_{\{\tau_H>T\}}
\right].
$$

Equivalently, the price solves the Black–Scholes PDE inside the domain below
the barrier,

$$
\frac{\partial V}{\partial t}
+\frac{1}{2}\sigma^2S^2\frac{\partial^2V}{\partial S^2}
+(r-q)S\frac{\partial V}{\partial S}
-rV=0,
\qquad 0<S<H,
$$

with terminal and absorbing-boundary conditions

$$
V(S,T)=(S-K)^+,\qquad V(H,t)=0.
$$

The probabilistic derivation below constructs the transition density satisfying
that absorbing boundary.

## 2. Log-price process

Set

$$
X_t=\log S_t,\qquad x_0=\log S_0,\qquad h=\log H,
\qquad k=\log K.
$$

Then

$$
X_t=x_0+\nu t+\sigma W_t,
\qquad
\nu=r-q-\frac{1}{2}\sigma^2.
$$

It is convenient to introduce

$$
\mu=\frac{\nu}{\sigma^2}
=\frac{r-q}{\sigma^2}-\frac{1}{2},
\qquad
\delta=\sigma\sqrt{T}.
$$

These are the quantities named `mu` and `stddev` in the C++ implementation.
Without a barrier, the transition density of $X_T$ is

$$
p_\nu(T;x,y)
=\frac{1}{\sigma\sqrt{2\pi T}}
\exp\left[-\frac{(y-x-\nu T)^2}{2\sigma^2T}\right].
$$

## 3. Absorbing transition density by reflection

To enforce the upper absorbing boundary at $h$, reflect the initial log spot
across the barrier,

$$
x^\star=2h-x_0
=\log\left(\frac{H^2}{S_0}\right).
$$

For $y<h$, the killed transition density is

$$
p_H(T;x_0,y)
=p_\nu(T;x_0,y)
-\exp\left[\frac{2\nu(h-x_0)}{\sigma^2}\right]
 p_\nu(T;x^\star,y).
$$

Using $\mu=\nu/\sigma^2$, the image coefficient becomes

$$
\alpha
=\exp\left[2\mu(h-x_0)\right]
=\left(\frac{H}{S_0}\right)^{2\mu}.
$$

The coefficient is fixed by the absorbing boundary. In fact,

$$
p_\nu(T;x_0,h)
=\alpha\,p_\nu(T;x^\star,h),
$$

so $p_H(T;x_0,h)=0$. The subtraction removes paths that have crossed the
barrier before maturity, even when their terminal value lies below it.

The option price is now

$$
C_{\mathrm{UO}}
=e^{-rT}\int_k^h(e^y-K)p_H(T;x_0,y)\,dy.
$$

Substituting the killed density splits the price into a direct Gaussian term
and a reflected image term.

## 4. Gaussian integration identities

Let $Y\sim\mathcal{N}(m,v)$. Completing the square gives

$$
\int_a^\infty f_Y(y)\,dy
=\Phi\left(\frac{m-a}{\sqrt v}\right),
$$

and

$$
\int_a^\infty e^y f_Y(y)\,dy
=e^{m+v/2}
\Phi\left(\frac{m+v-a}{\sqrt v}\right),
$$

where $\Phi$ is the standard normal CDF. Corresponding lower-tail integrals are
obtained by replacing $\Phi(z)$ with $\Phi(-z)$ after reversing the standardized
limit. These identities evaluate both the direct and image integrals.

## 5. Direct contribution: `A - B`

Define

$$
x_1
=\frac{\log(S_0/K)}{\delta}+(1+\mu)\delta,
$$

$$
x_2
=\frac{\log(S_0/H)}{\delta}+(1+\mu)\delta.
$$

The integral of the ordinary lognormal density over $k<y<h$ can be written as
a tail above the strike minus a tail above the barrier:

$$
e^{-rT}\int_k^h(e^y-K)p_\nu(T;x_0,y)\,dy=A-B,
$$

where

$$
A
=S_0e^{-qT}\Phi(x_1)
-Ke^{-rT}\Phi(x_1-\delta),
$$

$$
B
=S_0e^{-qT}\Phi(x_2)
-Ke^{-rT}\Phi(x_2-\delta).
$$

`A` is the usual Black–Scholes call value. `B` removes the part of the direct
terminal density lying at or above the barrier. This terminal truncation alone
is not enough, because a path may cross $H$ and later finish below $H$; the image
term removes those paths.

## 6. Reflected contribution: `C - D`

The reflected starting level is

$$
S^\star=e^{x^\star}=\frac{H^2}{S_0}.
$$

Define

$$
y_1
=\frac{\log(H^2/(S_0K))}{\delta}+(1+\mu)\delta,
$$

$$
y_2
=\frac{\log(H/S_0)}{\delta}+(1+\mu)\delta.
$$

For an upper integration limit $z<h$, define the discounted, image-weighted
lower-tail payoff integral

$$
F(z)
=\alpha e^{-rT}
\int_{-\infty}^{z}(e^y-K)p_\nu(T;x^\star,y)\,dy.
$$

Evaluating it at the strike and barrier gives

$$
C=F(k)
=S_0e^{-qT}
\left(\frac{H}{S_0}\right)^{2(\mu+1)}
\Phi(-y_1)
-Ke^{-rT}
\left(\frac{H}{S_0}\right)^{2\mu}
\Phi[-(y_1-\delta)],
$$

$$
D=F(h)
=S_0e^{-qT}
\left(\frac{H}{S_0}\right)^{2(\mu+1)}
\Phi(-y_2)
-Ke^{-rT}
\left(\frac{H}{S_0}\right)^{2\mu}
\Phi[-(y_2-\delta)].
$$

The image integral over $k<y<h$ is $F(h)-F(k)=D-C$. Because the image density
is subtracted from the direct density, its contribution to the option price is

$$
-(D-C)=C-D.
$$

## 7. Closed-form up-and-out call price

Combining the direct and reflected pieces gives

$$
\boxed{
C_{\mathrm{UO}}=A-B+C-D
}
$$

for $S_0<H$, $K<H$, continuous monitoring, and zero rebate, with

$$
\mu=\frac{r-q}{\sigma^2}-\frac{1}{2},
\qquad
\delta=\sigma\sqrt{T},
$$

and $A$, $B$, $C$, and $D$ defined above.

The mathematical value is nonnegative. The implementation returns
`max(0.0, A - B + C - D)` to suppress a possible tiny negative value caused by
floating-point cancellation near degenerate parameter limits.

## 8. Mapping to `src/barrier_common.hpp`

The implementation follows the derivation directly:

| Mathematical quantity | C++ expression |
|---|---|
| $\delta$ | `stddev = volatility * sqrt(maturity)` |
| $\mu$ | `(rate - dividend_yield) / volatility^2 - 0.5` |
| $(1+\mu)\delta$ | `mu_sigma` |
| $A$ | `vanilla_piece(strike)` |
| $B$ | `vanilla_piece(barrier)` |
| $(H/S_0)^{2\mu}$ | `pow0` |
| $(H/S_0)^{2(\mu+1)}$ | `pow1` |
| $C$ | `reflected_piece(H^2 / (S0 * K))` |
| $D$ | `reflected_piece(H / S0)` |

The lambda `reflected_piece` receives the dimensionless ratio appearing inside
the logarithm. The comments `eta=-1, phi=+1` refer to the conventional general
barrier-option notation: an upper barrier has reflection sign $\eta=-1$, and a
call has payoff sign $\phi=+1$.

## 9. Numerical check for the default example

For

$$
S_0=100,\quad K=100,\quad H=130,\quad T=1,
\quad r=0.03,\quad q=0,\quad \sigma=0.20,
$$

the intermediate values are approximately

$$
\mu=0.25,\quad
x_1=0.25,\quad
x_2=-1.0618213,\quad
y_1=2.8736426,\quad
y_2=1.5618213.
$$

The four components are

$$
A=9.41340338,\quad
B=4.37109408,\quad
C=-0.02417206,\quad
D=1.81538757,
$$

and therefore

$$
C_{\mathrm{UO}}
=A-B+C-D
=3.20274968.
$$

This is the analytic value shown in the repository's saved comparison plots.

## 10. Relation to the Brownian-bridge Monte Carlo correction

The analytic formula monitors the barrier continuously. A time-discretized
Monte Carlo path can miss a crossing between two sampled endpoints. Conditional
on two log-price endpoints $x_i<h$ and $x_{i+1}<h$, the Brownian-bridge survival
probability over a step of length $\Delta t$ is

$$
1-\exp\left[
-\frac{2(h-x_i)(h-x_{i+1})}{\sigma^2\Delta t}
\right].
$$

This interval formula is another application of the same reflection principle.
Multiplying these conditional survival probabilities across time steps makes
the Monte Carlo estimator target the same continuously monitored contract as
the closed-form solution.