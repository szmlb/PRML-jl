include("Regression.jl")

using LinearArgebra
using Statistics

mutable struct LinearRegression <: Regression
    #Linear regression model
    #y = X * w
    #t ~ N(t|X * w, var)

    w::Array{Float64, 1}
    var::Float64
end

function fit(self::LinearRegression, X::Array{Float64, 2}, t::Array{Float64, 1})
	#perform least squares fitting
	#Parameters
	#X : NxD Array{Float64, 2}
	#    training independent variable
	#t : N-element Array{Float64, 1}
	#    training dependent variable

	self.w = X \ t
	self.var = mean((X * self.w - t).^2)
end

function predict(self::LinearRegression, X, return_std::Bool=false)
	#make prediction given input
	#Parameters
	#----------
	#X : NxD Array{Float64, 2}
	#    samples to predict their output
	#return_std : Bool, optional
	#    returns standard deviation of each predition if true
	#Returns
	#-------
	#y : N-element Array{Float64, 1}
	#    prediction of each sample
	#y_std : N-element Array{Float64, 1}
	#    standard deviation of each predition

	y = X * self.w
	if return_std = true:
	    y_std = self.var^2 + zeros(size(y))
	    return y, y_std
	return y
end
