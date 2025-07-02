#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <utility>
#include <limits>
#include <chrono>
#include <random>

using namespace std;

struct Point {
	double x, y;
	Point() : x(0), y(0) {}
	Point(double x, double y) : x(x), y(y) {}
	Point operator+(const Point& rhs) const {
		return Point(x + rhs.x, y + rhs.y);
	}
	Point operator-(const Point& rhs) const {
		return Point(x - rhs.x, y - rhs.y);
	}
	Point operator*(const double rhs) const {
		return Point(x * rhs, y * rhs);
	}
	Point operator/(const double rhs) const {
		return Point(x / rhs, y / rhs);
	}
	double operator*(const Point& rhs) const {
		return (x * rhs.x) + (y * rhs.y);
	}
	double cross(const Point& rhs) const {
		return (x * rhs.y) - (y * rhs.x);
	}
	double length() const {
		return sqrt(x*x + y*y);
	}
	Point perp() const {
		return Point(-y, x);
	}
	Point norm() const {
		return Point(x, y) / length();
	}
	bool operator<(const Point& rhs) const {
		if (x == rhs.x) return y < rhs.y;
		else return x < rhs.x;
	}
	bool operator>(const Point& rhs) const {
		if (x == rhs.x) return y > rhs.y;
		else return x > rhs.x;
	}
	bool operator==(const Point& rhs) const {
		return x == rhs.x && y == rhs.y;
	}
	bool operator!=(const Point& rhs) const {
		return x != rhs.x || y != rhs.y;
	}
	friend ostream& operator<<(ostream& os, const Point& rhs) {
		os << "(" << rhs.x << ", " << rhs.y << ")";
		return os;
	}
};

enum inter_t {parallel = -3, collinear = -2, projective = -1, real = 0, overlap = 1};
struct Segment {
	Point s, e;
	Segment(Point s, Point e) : s(s), e(e) {}
	Segment(double sx, double sy, double ex, double ey) : s(Point(sx, sy)), e(Point(ex, ey)) {}
	pair<Point, inter_t> intersection(const Segment& rhs) const {
		Segment l(min(s,e), max(s,e)), r(min(rhs.s,rhs.e), max(rhs.s,rhs.e));
		Point lse = l.e - l.s, rse = r.e - r.s;
		Point diff = l.s - r.s;
		if (lse.x * rse.y == rse.x * lse.y) {
			if (diff.x * rse.y == rse.x * diff.y) {
				if (l.e.x >= r.s.x && r.e.x >= l.s.x) {
					return pair<Point, inter_t>(max(l.s, r.s), overlap);
				} else {
					return pair<Point, inter_t>(max(l.s, r.s), collinear);
				}
			} else {
				return pair<Point, inter_t>(Point(), parallel);
			}
		} else {
			double lt = (l.s-r.s).cross(rse)/rse.cross(lse);
			double rt = (l.s-r.s).cross(lse)/rse.cross(lse);
			Point inter = max(l.s + lse * lt, r.s + rse * rt);
			bool proj = lt * rt > min(lt, rt) || lt + rt < max(lt, rt);
			if (proj) {
				return pair<Point, inter_t>(inter, projective);
			} else {
				return pair<Point, inter_t>(inter, real);
			}
		}
	}
	double length() const {
		return (e - s).length();
	}
	bool operator==(const Segment& rhs) const {
		return s == rhs.s && e == rhs.e;
	}
	friend ostream& operator<<(ostream& os, const Segment& rhs) {
		os << rhs.s << " - " << rhs.e;
		return os;
	}
};

struct PlayerController {
	virtual vector<double> tick(vector<double> state) = 0;
};

struct PongGame {
	const int tickrate = 60;	// For animation convenience
	int max_score = 1;	// Tracked stats and limit
	int left_score = 0, right_score = 0;
	int left_returns = 0, right_returns = 0;
	int left_shots = 0, right_shots = 0;
	const double length = 400, width = 300, paddle_width = width/8, paddle_max_vel = width/tickrate;	// Dimensions
	const Point ball_start_vel = Point(length/tickrate, length/tickrate);
	Point ball_pos = Point(0, 0), ball_vel = ball_start_vel;	// Ball state
	double left_pos = 0, left_vel = 0;	// Player states
	double right_pos = 0, right_vel = 0;
	PlayerController & left, & right;	// Player controllers
	bool enable_random = true;
	default_random_engine generator;	// RNG for slight random deflection off paddles
	normal_distribution<double> distribution;
	PongGame(PlayerController & left, PlayerController & right) : left(left), right(right) {
		generator = default_random_engine(chrono::system_clock::now().time_since_epoch().count());
		distribution = normal_distribution<double>(0.0, 0.05);
	}
	void tick() {
		// Get left velocity from controller
		double left_cont = left.tick(vector<double>({
			2*ball_pos.x/length, 2*ball_pos.y/width,
			2*ball_vel.x/length, 2*ball_vel.y/width,
			2*left_pos/width, 2*left_vel/width,
			2*right_pos/width, 2*right_vel/width
		})).front() * paddle_max_vel;

		// Get right velocity from controller
		double right_cont = right.tick(vector<double>({
			-2*ball_pos.x/length, -2*ball_pos.y/width,
			-2*ball_vel.x/length, -2*ball_vel.y/width,
			-2*right_pos/width, -2*right_vel/width,
			-2*left_pos/width, -2*left_vel/width
		})).front() * -1 * paddle_max_vel;

		// Update paddle positions and velocities
		left_vel = left_cont;
		left_pos += left_vel;
		if (left_pos < paddle_width/2 - width/2) {
			left_pos = paddle_width/2 - width/2;
			left_vel = 0;
		}
		if (left_pos > width/2 - paddle_width/2) {
			left_pos = width/2 - paddle_width/2;
			left_vel = 0;
		}

		right_vel = right_cont;
		right_pos += right_vel;
		if (right_pos < paddle_width/2 - width/2) {
			right_pos = paddle_width/2 - width/2;
			right_vel = 0;
		}
		if (right_pos > width/2 - paddle_width/2) {
			right_pos = width/2 - paddle_width/2;
			right_vel = 0;
		}

		// Prepare geometry segments
		Segment mvmt(ball_pos, ball_pos + ball_vel);
		Segment upr_wall(Point(-length/2, -width/2), Point(length/2, -width/2));
		Segment lwr_wall(Point(-length/2, width/2), Point(length/2, width/2));
		Segment left_seg(Point(-length/2, left_pos - paddle_width/2), Point(-length/2, left_pos + paddle_width/2));
		Segment right_seg(Point(length/2, right_pos - paddle_width/2), Point(length/2, right_pos + paddle_width/2));

		// Do all collisions with segments
		vector<Segment> collidees = {upr_wall, lwr_wall, left_seg, right_seg};
		for (int hit = 0; hit >= 0;) {
			hit = -1;
			double best = numeric_limits<double>::max();	// Find closest collision first
			for (int i = 0; i < collidees.size(); ++i) {
				auto inter = mvmt.intersection(collidees[i]);
				if (inter.second >= real && inter.first != mvmt.s) {
					if ((inter.first - mvmt.s).length() < best) {
						best = (inter.first - mvmt.s).length();
						hit = i;
					}
				}
			}
			if (hit >= 0) {	// Handle hit, including left and right paddle special conditions
				Segment seg = collidees[hit];
				collidees.erase(collidees.begin() + hit);
				if (seg == left_seg) ++left_returns;
				if (seg == right_seg) ++right_returns;
				auto inter = mvmt.intersection(seg);
				mvmt.s = inter.first;
				Point perpv = (seg.e - seg.s).perp().norm();
				mvmt.e = mvmt.e - perpv * 2.0 * (perpv * (mvmt.e - mvmt.s));
				ball_vel = ball_vel - perpv * 2.0 * (perpv * ball_vel);
				if (seg == left_seg) ball_vel.y += left_vel;
				if (seg == right_seg) ball_vel.y += right_vel;
				if (enable_random && (seg == left_seg || seg == right_seg)) ball_vel.y += distribution(generator) * ball_start_vel.y;
			}
		}

		ball_pos = mvmt.e;	// Update ball position

		// Update shot statistics
		// A shot is a tick in which the ball is headed to score
		if (ball_vel.x < 0) {
			double shotpos = ball_pos.y + (-length/2 - ball_pos.x) * ball_vel.y / ball_vel.x;
			if (-width/2 <= shotpos && shotpos <= width/2)
				if (left_pos - paddle_width/2 > shotpos || left_pos + paddle_width/2 < shotpos)
					++right_shots;
		} else if (ball_vel.x > 0) {
			double shotpos = ball_pos.y + (length/2 - ball_pos.x) * ball_vel.y / ball_vel.x;
			if (-width/2 <= shotpos && shotpos <= width/2)
				if (right_pos - paddle_width/2 > shotpos || right_pos + paddle_width/2 < shotpos)
					++left_shots;
		}

		// If a point has been scored, process it and reset state as appropriate
		if (abs(ball_pos.x) > length/2) {
			if (ball_pos.x < 0) {
				++right_score;
				ball_vel = ball_start_vel * -1;
			} else {
				++left_score;
				ball_vel = ball_start_vel;
			}
			ball_pos = Point(0, 0);
			left_pos = 0;
			right_pos = 0;
		}

		assert(abs(ball_pos.y) <= width/2); // We should now still be inside bounds
	}
	// Run simulation until score or timelimit
	pair<int, int> simulate() {
		int timelimit = 2 * 16 * max_score * length / abs(ball_start_vel.x);
		while (max(left_score, right_score) < max_score && timelimit > 0) {
			tick();
			--timelimit;
		}
		return pair<int, int>(left_score, right_score);
	}
};