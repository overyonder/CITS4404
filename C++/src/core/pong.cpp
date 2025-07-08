#include "pong.hpp"
#include <cassert>
#include <cmath>
#include <limits>
#include <algorithm>
#include <chrono>

PongGame::PongGame(PlayerController& left, PlayerController& right) 
    : left(left), right(right) {
    generator = std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count());
    distribution = std::normal_distribution<double>(0.0, 0.05);
}

void PongGame::tick() {
    // Get left velocity from controller
    double left_cont = left.tick(std::vector<double>({
        2*ball_pos.x/length, 2*ball_pos.y/width,
        2*ball_vel.x/length, 2*ball_vel.y/width,
        2*left_pos/width, 2*left_vel/width,
        2*right_pos/width, 2*right_vel/width
    })).front() * paddle_max_vel;

    // Get right velocity from controller
    double right_cont = right.tick(std::vector<double>({
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
    std::vector<Segment> collidees = {upr_wall, lwr_wall, left_seg, right_seg};
    for (int hit = 0; hit >= 0;) {
        hit = -1;
        double best = std::numeric_limits<double>::max();	// Find closest collision first
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
            if (enable_random && (seg == left_seg || seg == right_seg)) 
                ball_vel.y += distribution(generator) * ball_start_vel.y;
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
    if (std::abs(ball_pos.x) > length/2) {
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

    assert(std::abs(ball_pos.y) <= width/2); // We should now still be inside bounds
}

std::pair<int, int> PongGame::simulate() {
    int timelimit = 2 * 16 * max_score * length / std::abs(ball_start_vel.x);
    while (std::max(left_score, right_score) < max_score && timelimit > 0) {
        tick();
        --timelimit;
    }
    return std::pair<int, int>(left_score, right_score);
} 