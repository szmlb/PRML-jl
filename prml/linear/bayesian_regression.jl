#include("regression.jl")

using LinearAlgebra
using Statistics

#mutable struct BayesianRegression <: Regression
mutable struct BayesianRegression
	#=
	Bayesian regression model

	w ~ N(w|0, alpha^(-1)I)
	y = X * w
	t ~ N(t|X * w, beta^(-1))
	=#

	alpha::Float64
	beta::Float64
 	w_mean
 	w_precision
	w_cov

	BayesianRegression(alpha, beta) = new(alpha, beta)

end

function _is_prior_defined(self::BayesianRegression)
	try 
		w_mean = self.w_mean
	catch e
		if isa(e, UndefRefError)
			self.w_mean = undef
		end
	end
	try 
		w_precision = self.w_precision
	catch e
		if isa(e, UndefRefError)
			self.w_precision = undef
		end
	end

        return self.w_mean != undef && self.w_precision != undef
end

function _get_prior(self::BayesianRegression, ndim)
	if _is_prior_defined(self)
		return self.w_mean, self.w_precision
	else
		return zeros(ndim), self.alpha * Matrix{Float64}(I, ndim, ndim)
	end
end

function fit(self::BayesianRegression, X, t)
	#=
	bayesian update of parameters given training dataset

	Parameters
	X : NxD Array{Float64, 2}
	    training independent variable
	t : N-element Array{Float64, 1}
	    training dependent variable
	=#

	mean_prev, precision_prev = _get_prior(self, size(X)[2])
		
	w_precision = precision_prev + self.beta * X' * X
	w_mean = (w_precision) \ (precision_prev * mean_prev + self.beta * X' * t)
	self.w_mean = w_mean
	self.w_precision = w_precision
	self.w_cov = inv(self.w_precision)

end

function predict(self::BayesianRegression, X, return_std::Bool=false, sample_size=undef)
	#=
	return mean (and standard deviation) of predictive distribution

	Parameters
	----------
	X : NxD Array{Float64, 2}
	    samples to predict their output
	return_std : Bool, optional
	    returns standard deviation of each predition if true
	sample_size : int, optional
	    number of samples to draw from the predictive distribution
	    (the default is None, no sampling from the distribution)

	Returns
	-------
	y : N-element Array{Float64, 1}
	    prediction of each sample
	y_std : N-element Array{Float64, 1}
	    standard deviation of each predition
	y_sample : Nxsample_size Array{Float64, 2}
	    samples from the predictive distribution
	=#

        if sample_size != undef
            w_sample = rand(MvNormal(self.w_mean, self.w_cov, sample_size))
            y_sample = X * w_sample'
            return y_sample
	end
        y = X * self.w_mean
        if return_std
            y_var = 1 / self.beta .+ sum(X * self.w_cov .* X, dims=2)
            y_std = y_var.^(1/2)
            return y, y_std
	end
        return y

end
