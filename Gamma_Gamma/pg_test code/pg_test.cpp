#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <boost/math/special_functions/digamma.hpp>
#include <fstream>
#include <random>
#include <numeric>

using std::vector;


// Negative log-likelihood for Poisson-Gamma model
double NegLogLik(
    const std::vector<double>& y, // Observed counts
    const std::vector<double>& eta, // Fixed effects F(X_ij)
    const std::vector<int>& group, 
    double gamma)
{
    std::unordered_map<int, double> S;           // sum y per group
    std::unordered_map<int, double> T;           // sum exp(eta) per group
    std::unordered_map<int, double> sum_eta_y;   // sum eta*y per group
    std::unordered_map<int, double> sum_logfact; // sum log(y!) per group

    const int n = y.size();

    for (int i = 0; i < n; ++i) {
        int g = group[i];
        S[g] += y[i];
        T[g] += std::exp(eta[i]);
        sum_eta_y[g] += eta[i] * y[i];
        sum_logfact[g] += std::lgamma(y[i] + 1.0);
    }

    double nll = 0.0;

    for (auto& kv : S) {
        int g = kv.first;

        double Si = S[g];
        double Ti = T[g];

        double gamma_prime = gamma + Si;
        double lambda_prime = gamma + Ti;

        nll += - sum_eta_y[g]
               + sum_logfact[g]
               - std::lgamma(gamma_prime)
               + gamma_prime * std::log(lambda_prime)
               - gamma * std::log(gamma)
               + std::lgamma(gamma);
    }

    return nll;
}

/* ============================
   Gradient w.r.t. F (eta)
   ============================ */

vector<double> GradF(
    const vector<double>& y,
    const vector<double>& eta,
    const vector<int>& group,
    double gamma)
{
    size_t n = y.size();
    vector<double> grad(n);

    std::unordered_map<int,double> S,T;

    for (size_t i=0;i<n;++i){
        int g = group[i];
        S[g] += y[i];
        T[g] += std::exp(eta[i]);
    }

    for (size_t i=0;i<n;++i){
        int g = group[i];
        double c = (gamma + S[g]) / (gamma + T[g]);
        grad[i] = -y[i] + c * std::exp(eta[i]);
    }

    return grad;
}

/* ============================
   Gradient w.r.t. beta
   ============================ */

vector<double> GradBeta(
    const vector<double>& y,
    const vector<double>& eta,
    const vector<int>& group,
    const vector<vector<double>>& X,
    double gamma)
{
    size_t n = y.size();
    size_t p = X[0].size();
    vector<double> grad_beta(p,0.0);

    vector<double> gradF = GradF(y,eta,group,gamma);

    for (size_t i=0;i<n;++i){
        for (size_t j=0;j<p;++j){
            grad_beta[j] += X[i][j] * gradF[i];
        }
    }

    return grad_beta;
}

/* ============================
   Gradient w.r.t. gamma
   ============================ */

double GradGamma(
    const vector<double>& y,
    const vector<double>& eta,
    const vector<int>& group,
    double gamma)
{
    std::unordered_map<int,double> S,T;

    for (size_t i=0;i<y.size();++i){
        int g = group[i];
        S[g] += y[i];
        T[g] += std::exp(eta[i]);
    }

    double grad = 0.0;

    for (auto &kv : S){
        int g = kv.first;
        double Si = S[g];
        double Ti = T[g];

        double gamma_p = gamma + Si;
        double lambda_p = gamma + Ti;

        grad += - boost::math::digamma(gamma_p)
                + std::log(lambda_p)
                + gamma_p / lambda_p
                - std::log(gamma)
                - 1.0
                + boost::math::digamma(gamma);
    }

    return grad;
}



/* ============================
   Linear Boosting
   ============================ */

void FitLinearPG(
    const vector<vector<double>>& X,
    const vector<double>& y,
    const vector<int>& group,
    vector<double>& beta,
    double& gamma,
    double learning_rate,
    int M)
{
    size_t n = y.size();
    size_t p = beta.size();

    vector<double> eta(n);

    for(int m=0; m<M; ++m){

        // compute eta = X beta
        for(size_t i=0;i<n;++i){
            eta[i] = 0.0;
            for(size_t j=0;j<p;++j){
                eta[i] += X[i][j]*beta[j];
            }
        }

        // compute gradients
        vector<double> grad_beta = GradBeta(y,eta,group,X,gamma);
        double grad_gamma = GradGamma(y,eta,group,gamma);

        // gradient descent step
        for(size_t j=0;j<p;++j){
            beta[j] -= learning_rate * grad_beta[j];
        }

        gamma -= learning_rate * grad_gamma;

        if(gamma <= 1e-6)
            gamma = 1e-6;  // enforce positivity

        double nll = NegLogLik(y,eta,group,gamma);

        std::cout << "Iter " << m
                  << " NLL: " << nll
                  << " gamma: " << gamma
                  << std::endl;
    }
}




/* ============================
   Linear Boosting with Plot 
   ============================ */

void FitLinearPG_plot(
    const vector<vector<double>>& X,
    const vector<double>& y,
    const vector<int>& group,
    vector<double>& beta,
    double& gamma,
    double learning_rate,
    int M)
{
    size_t n = y.size();
    size_t p = beta.size();

    std::vector<double> nll_values;
    std::vector<double> gamma_values;

    vector<double> eta(n);

    for(int m=0; m<M; ++m){

        for(size_t i=0;i<n;++i){
            eta[i] = 0.0;
            for(size_t j=0;j<p;++j){
                eta[i] += X[i][j]*beta[j];
            }
        }

        double nll = NegLogLik(y,eta,group,gamma);

        nll_values.push_back(nll);
        gamma_values.push_back(gamma);

        vector<double> grad_beta = GradBeta(y,eta,group,X,gamma);
        double grad_gamma = GradGamma(y,eta,group,gamma);

        for(size_t j=0;j<p;++j)
            beta[j] -= learning_rate * grad_beta[j];

        gamma -= learning_rate * grad_gamma;

        if(gamma <= 1e-6)
            gamma = 1e-6;
    }


    std::ofstream file("pg_training_trace.csv");

    file << "iter,nll,gamma\n";

    for(size_t i=0;i<nll_values.size();++i){
        file << i << ","
            << nll_values[i] << ","
            << gamma_values[i] << "\n";
    }

    file.close();
}




int main(){

    const int n_sim = 100;
    const int I = 100;
    const int n_i = 10;
    const int p = 10;

    const double true_gamma = 0.5;

    std::vector<double> true_beta(p,1.0);
    true_beta[0] = 0.0;

    std::mt19937 rng(123);

    std::vector<double> gamma_estimates;
    std::vector<std::vector<double>> beta_estimates;

    for(int sim=0; sim<n_sim; ++sim){

        int n = I * n_i;

        std::vector<int> group(n);
        for(int i=0;i<I;++i){
            for(int j=0;j<n_i;++j){
                group[i*n_i + j] = i;
            }
        }

        // Simulate random effects
        std::gamma_distribution<double> gamma_dist(true_gamma, 1.0/true_gamma);
        std::vector<double> theta(I);
        for(int i=0;i<I;++i)
            theta[i] = gamma_dist(rng);

        // Simulate X
        std::uniform_real_distribution<double> unif(-0.5,0.5);

        std::vector<std::vector<double>> X(n,std::vector<double>(p));
        for(int i=0;i<n;++i){
            X[i][0] = 1.0;
            for(int j=1;j<p;++j)
                X[i][j] = unif(rng);
        }

        // Compute true F
        std::vector<double> F(n,0.0);
        for(int i=0;i<n;++i){
            for(int j=0;j<p;++j)
                F[i] += X[i][j] * true_beta[j];
        }

        // Simulate y
        std::vector<double> y(n);
        for(int i=0;i<n;++i){
            double mu = std::exp(F[i]) * theta[group[i]];
            std::poisson_distribution<int> pois(mu);
            y[i] = pois(rng);
        }

        // Initialise parameters
        std::vector<double> beta(p,0.0);
        double gamma = 1.0;
        double learning_rate = 0.001;
        int M = 500;

        std::vector<double> eta(n);

        for(int iter=0; iter<M; ++iter){

            // compute eta = X beta
            for(int i=0;i<n;++i){
                eta[i] = 0.0;
                for(int j=0;j<p;++j)
                    eta[i] += X[i][j] * beta[j];
            }

            auto grad_beta = GradBeta(y,eta,group,X,gamma);
            double grad_gamma = GradGamma(y,eta,group,gamma);

            // update beta
            for(int j=0;j<p;++j)
                beta[j] -= learning_rate * grad_beta[j];

            // log-gamma update
            double gamma_log = std::log(gamma);
            gamma_log -= learning_rate * gamma * grad_gamma;
            gamma = std::exp(gamma_log);
        }

        gamma_estimates.push_back(gamma);
        beta_estimates.push_back(beta);

        std::cout << "Simulation " << sim+1
                  << " gamma_hat = " << gamma << std::endl;
    }
    std::ofstream gamma_file("gamma_estimates.csv");

    gamma_file << "gamma\n";

    for(double g : gamma_estimates){
        gamma_file << g << "\n";
    }

    gamma_file.close();
    // Summary
    double mean_gamma = std::accumulate(gamma_estimates.begin(),
                                        gamma_estimates.end(),0.0) / n_sim;

    double sq=0.0;
    for(auto g: gamma_estimates)
        sq += (g-mean_gamma)*(g-mean_gamma);

    double sd_gamma = std::sqrt(sq/n_sim);

    std::cout << "\nTrue gamma: " << true_gamma << std::endl;
    std::cout << "Mean estimated gamma: " << mean_gamma << std::endl;
    std::cout << "Std gamma: " << sd_gamma << std::endl;

    return 0;
}


// int main(){

//     vector<double> y = {3,1,2,4};
//     vector<int> group = {0,0,1,1};

//     vector<vector<double>> X = {
//         {1.0,0.5},
//         {1.0,-0.3},
//         {1.0,0.7},
//         {1.0,0.2}
//     };

//     size_t p = X[0].size();

//     vector<double> beta(p, 0.0);
//     double gamma = 2.0;

//     FitLinearPG_plot(X,y,group,beta,gamma,0.01,10000);

//     return 0;
// }




// int main(){

//     vector<double> y = {3,1,2,4};
//     vector<double> eta = {0.2,-0.1,0.3,0.0};
//     vector<int> group = {0,0,1,1};

//     vector<vector<double>> X = {
//         {1.0,0.5},
//         {1.0,-0.3},
//         {1.0,0.7},
//         {1.0,0.2}
//     };

//     double gamma = 2.0;

//     auto gradF = GradF(y,eta,group,gamma);

//     double eps = 1e-6;

//     // numeric check for first eta
//     vector<double> eta_eps = eta;
//     eta_eps[0] += eps;

//     double nll_plus = NegLogLik(y,eta_eps,group,gamma);
//     double nll = NegLogLik(y,eta,group,gamma);

//     double num_grad = (nll_plus - nll)/eps;

//     std::cout << "Analytic gradF[0]: " << gradF[0] << std::endl;
//     std::cout << "Numeric gradF[0]:  " << num_grad << std::endl;



//     auto grad_beta = GradBeta(y,eta,group,X,gamma);


//     for(size_t k=0;k<grad_beta.size();++k){

//         // perturb beta_k
//         vector<vector<double>> X_eps = X;
//         vector<double> eta_eps = eta;

//         for(size_t i=0;i<eta.size();++i){
//             eta_eps[i] += eps * X[i][k];
//         }

//         double nll_plus = NegLogLik(y,eta_eps,group,gamma);
//         double nll = NegLogLik(y,eta,group,gamma);

//         double num_grad = (nll_plus - nll)/eps;

//         std::cout << "Beta " << k
//                 << " analytic: " << grad_beta[k]
//                 << " numeric: " << num_grad << std::endl;
//     }
// }





// int main()
// {
//     // Toy example
//     std::vector<double> y     = {3, 1, 2, 4};
//     std::vector<double> eta   = {0.2, -0.1, 0.3, 0.0};
//     std::vector<int> group    = {0, 0, 1, 1};

//     double gamma = 2.0;

//     double nll = NegLogLik_PoissonGamma(y, eta, group, gamma);

//     std::cout << "Negative log-likelihood = " << nll << std::endl;

//     return 0;
// }