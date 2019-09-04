using Combinatorics

mutable struct PolynomialFeature
	#=
	polynomial features
	transforms input array with polynomial features
	Example
	=======
	x =
	[a b;
	 c d]
	y = PolynomialFeatures(degree=2).transform(x)
	y =
	[1 a b a^2 a*b b^2;
	 1 c d c^2 c*d d^2]
	=#

	degree::Int64

	function PolynomialFeature(degree)
		#=
		construct polynomial features
		Parameters
		----------
		degree : Int64
		    degree of polynomial
		=#

		@assert isinteger(degree) "Not an integer number"
		new(degree)

	end

end

function transform(self::PolynomialFeature, x)
	#=
	transforms input array with polynomial features
	Parameters
	----------
	x : (sample_size)x(n) Array{Float64, 2}
	    input array
	Returns
	-------
	output : (sample_size)x(1 + nC1 + ... + nCd) Array{Float64, 2}
	    polynomial features
	=#

	sample_size = size(x)[1]
	features = []
	for sample in 1:sample_size
		features_tmp = [1.0]	
		for degree in 1:self.degree
			for items in with_replacement_combinations(x[sample, :], degree)
				append!(features_tmp, prod(items))
			end
		end
		if isempty(features) == true
			features = features_tmp'	
		else
			features = vcat(features, features_tmp')
		end
	end

	return features
end
