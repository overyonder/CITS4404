#include "geometry.hpp"
#include <cmath>
#include <algorithm>

// Point implementations
Point::Point() : x(0), y(0) {}

Point::Point(double x, double y) : x(x), y(y) {}

Point Point::operator+(const Point& rhs) const {
    return Point(x + rhs.x, y + rhs.y);
}

Point Point::operator-(const Point& rhs) const {
    return Point(x - rhs.x, y - rhs.y);
}

Point Point::operator*(const double rhs) const {
    return Point(x * rhs, y * rhs);
}

Point Point::operator/(const double rhs) const {
    return Point(x / rhs, y / rhs);
}

double Point::operator*(const Point& rhs) const {
    return (x * rhs.x) + (y * rhs.y);
}

double Point::cross(const Point& rhs) const {
    return (x * rhs.y) - (y * rhs.x);
}

double Point::length() const {
    return sqrt(x*x + y*y);
}

Point Point::perp() const {
    return Point(-y, x);
}

Point Point::norm() const {
    return Point(x, y) / length();
}

bool Point::operator<(const Point& rhs) const {
    if (x == rhs.x) return y < rhs.y;
    else return x < rhs.x;
}

bool Point::operator>(const Point& rhs) const {
    if (x == rhs.x) return y > rhs.y;
    else return x > rhs.x;
}

bool Point::operator==(const Point& rhs) const {
    return x == rhs.x && y == rhs.y;
}

bool Point::operator!=(const Point& rhs) const {
    return x != rhs.x || y != rhs.y;
}

std::ostream& operator<<(std::ostream& os, const Point& rhs) {
    os << "(" << rhs.x << ", " << rhs.y << ")";
    return os;
}

// Segment implementations
Segment::Segment(Point s, Point e) : s(s), e(e) {}

Segment::Segment(double sx, double sy, double ex, double ey) : s(Point(sx, sy)), e(Point(ex, ey)) {}

std::pair<Point, inter_t> Segment::intersection(const Segment& rhs) const {
    Segment l(std::min(s,e), std::max(s,e)), r(std::min(rhs.s,rhs.e), std::max(rhs.s,rhs.e));
    Point lse = l.e - l.s, rse = r.e - r.s;
    Point diff = l.s - r.s;
    if (lse.x * rse.y == rse.x * lse.y) {
        if (diff.x * rse.y == rse.x * diff.y) {
            if (l.e.x >= r.s.x && r.e.x >= l.s.x) {
                return std::pair<Point, inter_t>(std::max(l.s, r.s), overlap);
            } else {
                return std::pair<Point, inter_t>(std::max(l.s, r.s), collinear);
            }
        } else {
            return std::pair<Point, inter_t>(Point(), parallel);
        }
    } else {
        double lt = (l.s-r.s).cross(rse)/rse.cross(lse);
        double rt = (l.s-r.s).cross(lse)/rse.cross(lse);
        Point inter = std::max(l.s + lse * lt, r.s + rse * rt);
        bool proj = lt * rt > std::min(lt, rt) || lt + rt < std::max(lt, rt);
        if (proj) {
            return std::pair<Point, inter_t>(inter, projective);
        } else {
            return std::pair<Point, inter_t>(inter, real);
        }
    }
}

double Segment::length() const {
    return (e - s).length();
}

bool Segment::operator==(const Segment& rhs) const {
    return s == rhs.s && e == rhs.e;
}

std::ostream& operator<<(std::ostream& os, const Segment& rhs) {
    os << rhs.s << " - " << rhs.e;
    return os;
} 