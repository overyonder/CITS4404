#pragma once

#include <iostream>
#include <utility>

/**
 * 2D Point class with basic geometric operations
 */
struct Point {
    double x, y;
    
    Point();
    Point(double x, double y);
    
    // Arithmetic operators
    Point operator+(const Point& rhs) const;
    Point operator-(const Point& rhs) const;
    Point operator*(const double rhs) const;
    Point operator/(const double rhs) const;
    
    // Dot product
    double operator*(const Point& rhs) const;
    
    // Cross product (returns scalar for 2D)
    double cross(const Point& rhs) const;
    
    // Geometric operations
    double length() const;
    Point perp() const;        // Perpendicular vector
    Point norm() const;        // Normalized vector
    
    // Comparison operators
    bool operator<(const Point& rhs) const;
    bool operator>(const Point& rhs) const;
    bool operator==(const Point& rhs) const;
    bool operator!=(const Point& rhs) const;
    
    // Stream output
    friend std::ostream& operator<<(std::ostream& os, const Point& rhs);
};

/**
 * Line segment intersection types
 */
enum inter_t {
    parallel = -3,
    collinear = -2,
    projective = -1,
    real = 0,
    overlap = 1
};

/**
 * Line segment class with intersection calculation
 */
struct Segment {
    Point s, e;  // start and end points
    
    Segment(Point s, Point e);
    Segment(double sx, double sy, double ex, double ey);
    
    // Calculate intersection with another segment
    std::pair<Point, inter_t> intersection(const Segment& rhs) const;
    
    double length() const;
    
    bool operator==(const Segment& rhs) const;
    
    friend std::ostream& operator<<(std::ostream& os, const Segment& rhs);
}; 