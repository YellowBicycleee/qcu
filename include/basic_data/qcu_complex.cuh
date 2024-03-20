#pragma once

#include <cstdio>

// template <typename _Float>
class Complex
{
private:
    typedef double _Float;
    _Float real_;
    _Float imag_;

public:
    // constructors
    __device__ __host__ __forceinline__
    Complex(const double2 &rhs) : real_(rhs.x), imag_(rhs.y) {}
    __device__ __host__ __forceinline__ Complex(_Float real, _Float imag) : real_(real), imag_(imag) {}
    Complex() = default;
    __device__ __host__ __forceinline__
    Complex(const Complex &complex) : real_(complex.real_), imag_(complex.imag_) {}
    __device__ __host__ __forceinline__ 
    Complex(const _Float &rhs) : real_(rhs), imag_(0) {}

    __device__ __host__ __forceinline__ _Float norm2() { return sqrt(real_ * real_ + imag_ * imag_); }
    __device__ __host__ __forceinline__ void setImag(_Float imag) { imag_ = imag; }
    __device__ __host__ __forceinline__ void setReal(_Float real) { real_ = real; }
    __device__ __host__ __forceinline__ _Float real() const { return real_; }
    __device__ __host__ __forceinline__ _Float imag() const { return imag_; }

    __device__ __host__ __forceinline__ Complex &operator=(const Complex &complex)
    {
        real_ = complex.real_;
        imag_ = complex.imag_;
        return *this;
    }
    __device__ __host__ __forceinline__ Complex &operator=(_Float rhs)
    {
        real_ = rhs;
        imag_ = 0;
        return *this;
    }
    __device__ __host__ __forceinline__ Complex operator+(const Complex &complex) const
    {
        return Complex(real_ + complex.real_, imag_ + complex.imag_);
    }
    __device__ __host__ __forceinline__ Complex operator-(const Complex &complex) const
    {
        return Complex(real_ - complex.real_, imag_ - complex.imag_);
    }
    __device__ __host__ __forceinline__ Complex operator-() const { return Complex(-real_, -imag_); }
    __device__ __host__ __forceinline__ Complex operator*(const Complex &rhs) const
    {
        return Complex(real_ * rhs.real_ - imag_ * rhs.imag_, real_ * rhs.imag_ + imag_ * rhs.real_);
    }
    __device__ __host__ __forceinline__ Complex operator*(const _Float &rhs) const
    {
        return Complex(real_ * rhs, imag_ * rhs);
    }
    // __device__ __host__ Complex &operator*=(const Complex &rhs) // TO modify
    // {
    //   real_ = real_ * rhs.real_ - imag_ * rhs.imag_;
    //   imag_ = real_ * rhs.imag_ + imag_ * rhs.real_;
    //   return *this;
    // }
    __device__ __host__ __forceinline__ Complex &operator*=(const _Float &rhs)
    {
        real_ = real_ * rhs;
        imag_ = imag_ * rhs;
        return *this;
    }
    __device__ __host__ __forceinline__ Complex operator/(const _Float &rhs) { return Complex(real_ / rhs, imag_ / rhs); }
    __device__ __host__ __forceinline__ Complex operator/(const Complex &rhs) const
    {
        return (*this * rhs.conj()) / (rhs.real() * rhs.real() + rhs.imag() * rhs.imag());
    }
    __device__ __host__ __forceinline__ Complex &operator/=(const Complex &rhs)
    {
        _Float new_real = (real_ * rhs.real() + imag_ * rhs.imag()) / (rhs.real() * rhs.real() + rhs.imag() * rhs.imag());
        _Float new_imag = (rhs.real() * imag_ - real_ * rhs.imag()) / (rhs.real() * rhs.real() + rhs.imag() * rhs.imag());
        real_ = new_real;
        imag_ = new_imag;
        return *this;
    }
    __device__ __host__ __forceinline__ Complex &operator+=(const Complex &rhs)
    {
        real_ += rhs.real_;
        imag_ += rhs.imag_;
        return *this;
    }

    __device__ __host__ __forceinline__ Complex &operator-=(const Complex &rhs)
    {
        real_ -= rhs.real_;
        imag_ -= rhs.imag_;
        return *this;
    }

    __device__ __host__ __forceinline__ Complex &clear2Zero()
    {
        real_ = 0;
        imag_ = 0;
        return *this;
    }

    __device__ __host__ __forceinline__ Complex multiply_i()
    {
        return Complex(-imag_, real_);
    }
    __device__ __host__ __forceinline__ Complex multiply_minus_i()
    {
        return Complex(imag_, -real_);
    }
    __device__ __host__ __forceinline__ Complex &self_multiply_i()
    {
        _Float temp = real_;
        real_ = -imag_;
        imag_ = temp;
        return *this;
    }
    __device__ __host__ __forceinline__ Complex &self_multiply_minus_i()
    {
        _Float temp = -real_;
        real_ = imag_;
        imag_ = temp;
        return *this;
    }

    __device__ __host__ __forceinline__ Complex conj() const { return Complex(real_, -imag_); }
    __device__ __host__ __forceinline__ bool operator==(const Complex &rhs) { return real_ == rhs.real_ && imag_ == rhs.imag_; }
    __device__ __host__ __forceinline__ bool operator!=(const Complex &rhs) { return real_ != rhs.real_ || imag_ != rhs.imag_; }
    __device__ __host__ __forceinline__ void output() const { printf("(%lf + %lfi)", real_, imag_); }
};