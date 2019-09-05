#include("regression.jl")

using LinearAlgebra
using Statistics

#mutable struct RidgeRegression <: Regression
mutable struct RidgeRegression
	#=
	Ridge regression model

	w* = argmin |t - X * w| + alpha * |w|_2^2
	=#

 	w::Array{Float64, 1}
	alpha::Float64

	RidgeRegression(w, alpha) = new(w, alpha)

end

function fit(self::RidgeRegression, X, t)
	#=
	maximum a posteriori estimation of parameter

	Parameters
	X : NxD Array{Float64, 2}
	    training independent variable
	t : N-element Array{Float64, 1}
	    training dependent variable
	=#

	eye = Matrix{Float64}(I, size(X)[2], size(X)[2])
	self.w = (self.alpha * eye + X' * X) \ (X' * t)
end

function predict(self::RidgeRegression, X)
	#=
	make prediction given input

	Parameters
	----------
	X : NxD Array{Float64, 2}
	    samples to predict their output

	Returns
	-------
	y : N-element Array{Float64, 1}
	    prediction of each sample
	=#

	y = X * self.w
	return y
end
