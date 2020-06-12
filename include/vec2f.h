#pragma once

#include <math.h>
#include <assert.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

class Vec2f
{
public:
	union
	{
		float Element[2];
		struct { float X, Y; };
	};

public:
	CUDA_CALLABLE_MEMBER Vec2f() {}

	CUDA_CALLABLE_MEMBER Vec2f(float value)
		: X(value), Y(value) {}

	CUDA_CALLABLE_MEMBER Vec2f(float x, float y)
		: X(x), Y(y) {}

	CUDA_CALLABLE_MEMBER Vec2f(const Vec2f &p_vec)
		: X(p_vec.X), Y(p_vec.Y) {}

	CUDA_CALLABLE_MEMBER float operator[](int ind) const { return Element[ind]; }
	CUDA_CALLABLE_MEMBER float& operator[](int ind) { return Element[ind]; }

	CUDA_CALLABLE_MEMBER Vec2f& operator=(const Vec2f &p_vector)
	{
		X = p_vector.X;
		Y = p_vector.Y;
		
		return *this;
	}

	CUDA_CALLABLE_MEMBER inline Vec2f operator*(float val) const {
		return Vec2f(val * X, val * Y);
	}

	CUDA_CALLABLE_MEMBER inline Vec2f operator/(float val) const
	{
		assert(val != 0.0);
		return Vec2f(*this * (1.0f / val));
	}

	CUDA_CALLABLE_MEMBER inline Vec2f operator*(const Vec2f &p_vec) const {
		return Vec2f(p_vec.X * X, p_vec.Y * Y);
	}

	CUDA_CALLABLE_MEMBER inline Vec2f operator+(const Vec2f &p_vec) const {
		return Vec2f(X + p_vec.X, Y + p_vec.Y);
	}

	CUDA_CALLABLE_MEMBER inline Vec2f operator-(const Vec2f &p_vec) const {
		return Vec2f(X - p_vec.X, Y - p_vec.Y);
	}

	CUDA_CALLABLE_MEMBER inline Vec2f operator-(void) const {
		return Vec2f(-X, -Y);
	}

	CUDA_CALLABLE_MEMBER inline Vec2f& operator*=(float val) {
		return *this = *this * val;
	}

	CUDA_CALLABLE_MEMBER inline Vec2f& operator*=(const Vec2f &p_vec) {
		return *this = *this * p_vec;
	}

	CUDA_CALLABLE_MEMBER inline Vec2f& operator/=(float val) {
		return *this = *this / val;
	}

	CUDA_CALLABLE_MEMBER inline Vec2f& operator+=(const Vec2f &p_vec) {
		return *this = *this + p_vec;
	}
	
	CUDA_CALLABLE_MEMBER inline Vec2f& operator-=(const Vec2f &p_vec) {
		return *this = *this - p_vec;
	}

	CUDA_CALLABLE_MEMBER inline float length(void) const {
		return sqrt(X * X + Y * Y);
	}
};

CUDA_CALLABLE_MEMBER inline Vec2f operator*(float val, const Vec2f &p_vec) {
	return Vec2f(val * p_vec.X, val * p_vec.Y);
}
