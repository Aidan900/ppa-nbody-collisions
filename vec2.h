#pragma once

#include <math.h>
#include <assert.h>

template <class T>
class Vec2
{
public:
	union
	{
		T Element[2];
		struct { 
			T X;
			T Y; 
		};
	};

public:
	Vec2() {}

	Vec2(T value)
		: X(value), Y(value) {}

	Vec2(T x, T y)
		: X(x), Y(y) {}

	Vec2(const Vec2<T> &p_vec)
		: X(p_vec.X), Y(p_vec.Y) {}

	T operator[](int ind) const { return Element[ind]; }
	T& operator[](int ind) { return Element[ind]; }

	Vec2<T>& operator=(const Vec2<T> &p_vector)
	{
		X = p_vector.X;
		Y = p_vector.Y;
		
		return *this;
	}

	inline Vec2<T> operator*(T val) const {
		return Vec2<T>(val * X, val * Y);
	}

	inline Vec2<T> operator/(T val) const 
	{
		assert(val != 0.0);
		return Vec2<T>(*this * (1.0f / val));
	}

	inline Vec2<T> operator*(const Vec2<T> &p_vec) const {
		return Vec2<T>(p_vec.X * X, p_vec.Y * Y);
	}

	inline Vec2<T> operator+(const Vec2<T> &p_vec) const {
		return Vec2<T>(X + p_vec.X, Y + p_vec.Y);
	}

	inline Vec2<T> operator-(const Vec2<T> &p_vec) const {
		return Vec2<T>(X - p_vec.X, Y - p_vec.Y);
	}

	inline Vec2<T> operator-(void) const {
		return Vec2<T>(-X, -Y);
	}

	inline Vec2<T>& operator*=(T val) {
		return *this = *this * val;
	}

	inline Vec2<T>& operator*=(const Vec2<T> &p_vec) {
		return *this = *this * p_vec;
	}

	inline Vec2<T>& operator/=(T val) {
		return *this = *this / val;
	}

	inline Vec2<T>& operator+=(const Vec2<T> &p_vec) {
		return *this = *this + p_vec;
	}
	
	inline Vec2<T>& operator-=(const Vec2<T> &p_vec) {
		return *this = *this - p_vec;
	}

	inline T length(void) const {
		return sqrt(X * X + Y * Y);
	}
};

template <typename T>
inline Vec2<T> operator*(T val, const Vec2<T> &p_vec) {
	return Vec2<T>(val * p_vec.X, val * p_vec.Y);
}