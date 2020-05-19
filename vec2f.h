#pragma once

#include <math.h>
#include <assert.h>

class Vec2f
{
public:
	union
	{
		float Element[2];
		struct { double X, Y; };
	};

public:
	Vec2f() {}

	Vec2f(double value)
		: X(value), Y(value) {}

	Vec2f(double x, double y)
		: X(x), Y(y) {}

	Vec2f(const Vec2f &p_vec)
		: X(p_vec.X), Y(p_vec.Y) {}

	float operator[](int ind) const { return Element[ind]; }
	float& operator[](int ind) { return Element[ind]; }

	Vec2f& operator=(const Vec2f &p_vector)
	{
		X = p_vector.X;
		Y = p_vector.Y;
		
		return *this;
	}

	inline Vec2f operator*(double val) const {
		return Vec2f(val * X, val * Y);
	}

	inline Vec2f operator/(double val) const 
	{
		assert(val != 0.0);
		return Vec2f(*this * (1.0f / val));
	}

	inline Vec2f operator*(const Vec2f &p_vec) const {
		return Vec2f(p_vec.X * X, p_vec.Y * Y);
	}

	inline Vec2f operator+(const Vec2f &p_vec) const {
		return Vec2f(X + p_vec.X, Y + p_vec.Y);
	}

	inline Vec2f operator-(const Vec2f &p_vec) const {
		return Vec2f(X - p_vec.X, Y - p_vec.Y);
	}

	inline Vec2f operator-(void) const {
		return Vec2f(-X, -Y);
	}

	inline Vec2f& operator*=(double val) {
		return *this = *this * val;
	}

	inline Vec2f& operator*=(const Vec2f &p_vec) {
		return *this = *this * p_vec;
	}

	inline Vec2f& operator/=(double val) {
		return *this = *this / val;
	}

	inline Vec2f& operator+=(const Vec2f &p_vec) {
		return *this = *this + p_vec;
	}
	
	inline Vec2f& operator-=(const Vec2f &p_vec) {
		return *this = *this - p_vec;
	}

	inline float length(void) const {
		return sqrt(X * X + Y * Y);
	}
};

inline Vec2f operator*(double val, const Vec2f &p_vec) {
	return Vec2f(val * p_vec.X, val * p_vec.Y);
}