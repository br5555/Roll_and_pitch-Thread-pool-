#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <math.h>
#include <stdio.h>
#include <thread>
#include <iostream>
#include <fstream>
#include "Adam_MPC.h"
#include <chrono>
#include "ctpl.h" 
typedef std::chrono::high_resolution_clock Clock;
/*
const applies for variables, and prevents them from being modified in your code.

constexpr tells the compiler that this expression results in a compile time constant value,
so it can be used in places like array lengths, assigning to const variables, etc. The link given by Oli has a lot of excellent examples.
*/
constexpr int combined_control_mpc_use_ = 1;  // still working with moving masses

	// MM_MPC + added variables for CC_MPC
constexpr int kStateSize = 8;// 6 + 2 * combined_control_mpc_use_; // [x1, dx1, x3, dx3, theta, dtheta] -> A is [6,6]
constexpr int kInputSize = 4;// 2 + 2 * combined_control_mpc_use_; // [x1_ref (m), x3_ref (m)]          -> B is [6,2]
constexpr int kMeasurementSize = 1;                        // [theta] -> C is [1,6]
constexpr int kDisturbanceSize = kStateSize;               // disturbances are looked on all states -> B_d is [6,6]

constexpr int kControlHorizonSteps = 5;
constexpr int kPredictionHorizonSteps = 14;
constexpr double kGravity = 9.80665;


int main(int argc, char **argv) {
	int stop_display;

	Eigen::Matrix<double, kStateSize, 1> target_state_roll, target_state_pitch;
	Eigen::Matrix<double, kStateSize, 1> current_state_roll, current_state_pitch;
	Eigen::Matrix<double, kInputSize, 1> target_input_roll, target_input_pitch;
	Eigen::Matrix<double, kStateSize, kStateSize>       model_A_, model_A_70_ms;
	Eigen::Matrix<double, kStateSize, kInputSize>       model_B_, model_B_70_ms;
	Eigen::Matrix<double, kStateSize, kDisturbanceSize> model_Bd_;
	Eigen::Matrix<double, kInputSize, 1> r_command_;
	Eigen::Matrix<double, kDisturbanceSize, 1> estimated_disturbances_;
	Eigen::VectorXd ref(kMeasurementSize);
	Eigen::Matrix<double, kStateSize, kStateSize>       A_contin;
	Eigen::Matrix<double, kStateSize, kInputSize>       B_contin;
	Eigen::Matrix<double, kStateSize, kStateSize>       OV_scale;
	Eigen::Matrix<double, kInputSize, kInputSize>       MV_scale;
	Eigen::Matrix<double, kStateSize, kStateSize> Q;
	Eigen::Matrix<double, kStateSize, kStateSize> Q_final;
	Eigen::Matrix<double, kInputSize, kInputSize> R;
	Eigen::Matrix<double, kInputSize, kInputSize> R_delta;

	Eigen::Matrix<double, 2, 1>       u_max;
	Eigen::Matrix<double, 2, 1>       du_max;
	Eigen::Matrix<double, 2, 1>       u_min;
	Eigen::Matrix<double, 2, 1>       du_min;
	double         lm_ = 0.6;
	double sampling_time_(0.078684);


	current_state_pitch.setZero();
	current_state_pitch.setZero();

	u_max(0, 0) = lm_ / 2 - 0.01;

	u_min(0, 0) = -u_max(0, 0);



	du_max(0, 0) = sampling_time_ * 2;

	du_min(0, 0) = -du_max(0, 0);




	u_max(1, 0) = 50;

	u_min(1, 0) = -u_max(1, 0);



	du_max(1, 0) = 10;
	du_min(1, 0) = -10;





	ref << 0;
	r_command_ << 0, 0, 0, 0;

	MV_scale.setZero();
	OV_scale.setZero();
	OV_scale(0, 0) = 0.58;
	OV_scale(1, 1) = 4.0;
	OV_scale(2, 2) = 0.58;
	OV_scale(3, 3) = 4.0;
	OV_scale(4, 4) = 800;
	OV_scale(5, 5) = 800;
	OV_scale(6, 6) = 0.5236;
	OV_scale(7, 7) = 0.5236;

	MV_scale(0, 0) = 0.58;
	MV_scale(1, 1) = 0.58;
	MV_scale(2, 2) = 100;
	MV_scale(3, 3) = 100;




	Q_final.setIdentity();

	Q = Q_final;
	R.setIdentity();

	R_delta = R;


	R(0, 0) = 0.67670;
	R(1, 1) = 0.67670;
	R(2, 2) = 0.13534000;
	R(3, 3) = 0.13534000;

	R_delta(0, 0) = 0.738879858135067;
	R_delta(1, 1) = 0.738879858135067;
	R_delta(2, 2) = 0.007388798581351;
	R_delta(3, 3) = 0.007388798581351;



	Q(0, 0) = 0.135340000000000;
	Q(1, 1) = 0.002706800000000;
	Q(2, 2) = 0.1353400;
	Q(3, 3) = 0.002706800;
	Q(4, 4) = 0.002706800;
	Q(5, 5) = 0.002706800;
	Q(6, 6) = 10.7068000;
	Q(7, 7) = 9.676700;

	A_contin << 0, 1, 0, 0, 0, 0, 0, 0,
		-391.426, -25.894, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 0,
		0, 0, -391.426, -25.894, 0, 0, 0, 0,
		0, 0, 0, 0, -4, 0, 0, 0,
		0, 0, 0, 0, 0, -4, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 1,
		4.7025, 0.197406, 4.7025, 0.197406, 0.0417255, -0.0417255, 0, 0;

	B_contin << 0, 0, 0, 0,
		391.426, 0, 0, 0,
		0, 0, 0, 0,
		0, 391.426, 0, 0,
		0, 0, 4, 0,
		0, 0, 0, 4,
		0, 0, 0, 0,
		-2.98409, -2.98409, 0, 0;





	model_A_ << 0.782714, 0.0224343, 0, 0, 0, 0, 0, 0,
		-8.78138, 0.201802, 0, 0, 0, 0, 0, 0,
		0, 0, 0.782714, 0.0224343, 0, 0, 0, 0,
		0, 0, -8.78138, 0.201802, 0, 0, 0, 0,
		0, 0, 0, 0, 0.852144, 0, 0, 0,
		0, 0, 0, 0, 0, 0.852144, 0, 0,
		0.00297324, 0.000147926, 0.00297324, 0.000147926, 3.16691e-05, -3.16691e-05, 1, 0.04,
		0.130198, 0.00703908, 0.130198, 0.00703908, 0.00154234, -0.00154234, 0, 1;

	model_A_ << 1, 0.04, 0, 0, 0, 0, 0, 0,
		-15.6571, -0.0357581, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 0.04, 0, 0, 0, 0,
		0, 0, -15.6571, -0.0357581, 0, 0, 0, 0,
		0, 0, 0, 0, 0.84, 0, 0, 0,
		0, 0, 0, 0, 0, 0.84, 0, 0,
		0, 0, 0, 0, 0, 0, 1, 0.04,
		0.1881, 0.00789623, 0.1881, 0.00789623, 0.00166902, -0.00166902, 0, 1;

	model_B_ << 0.215525, 0, 0, 0,
		8.84393, 0, 0, 0,
		0, 0.215525, 0, 0,
		0, 8.84393, 0, 0,
		0, 0, 0.147975, 0,
		0, 0, 0, 0.147975,
		-0.00158618, -0.00158618, 1.68604e-06, -1.68604e-06,
		-0.0620126, -0.0620126, 0.000125442, -0.000125442;

	model_Bd_ << 0.0368517, 0.000550615, 0, 0, 0, 0, 0, 0,
		-0.215525, 0.0225941, 0, 0, 0, 0, 0, 0,
		0, 0, 0.0368517, 0.000550615, 0, 0, 0, 0,
		0, 0, -0.215525, 0.0225941, 0, 0, 0, 0,
		0, 0, 0, 0, 0.0369936, 0, 0, 0,
		0, 0, 0, 0, 0, 0.0369936, 0, 0,
		4.15876e-05, 1.98561e-06, 4.15876e-05, 1.98561e-06, 4.2151e-07, -4.2151e-07, 0.04, 0.000792,
		0.00294716, 0.000146518, 0.00294716, 0.000146518, 3.13605e-05, -3.13605e-05, 0, 0.04;

	estimated_disturbances_ << 0.0744648,
		-0.00684855,
		-0.468717,
		0.553529,
		20,
		20,
		-0.139467,
		-0.00190078;
	model_A_70_ms << 0.4271, 0.0223, 0, 0, 0, 0, 0, 0,
		-8.7243, -0.1501, 0, 0, 0, 0, 0, 0,
		0, 0, 0.4271, 0.0223, 0, 0, 0, 0,
		0, 0, -8.7243, -0.1501, 0, 0, 0, 0,
		0, 0, 0, 0, 0.7300, 0, 0, 0,
		0, 0, 0, 0, 0, 0.7300, 0, 0,
		0.0090, 0.0005, 0.0090, 0.0005, 0.0001, -0.0001, 1.0000, 0.0787,
		0.1699, 0.0113, 0.1699, 0.0113, 0.0028, -0.0028, 0, 1.0000;

	model_B_70_ms << 0.5729, 0, 0, 0,
		8.7243, 0, 0, 0,
		0, 0.5729, 0, 0,
		0, 8.7243, 0, 0,
		0, 0, 0.2700, 0,
		0, 0, 0, 0.2700,
		-0.0037, -0.0037, 0.0000, -0.0000,
		-0.0347, -0.0347, 0.0005, -0.0005;



	Adam_MPC roll_cost = Adam_MPC(model_A_70_ms, model_B_70_ms, model_Bd_, Q, Q_final, R, R_delta, estimated_disturbances_, kStateSize, 14, kControlHorizonSteps, MV_scale, OV_scale);
	Adam_MPC pitch_cost = Adam_MPC(model_A_70_ms, model_B_70_ms, model_Bd_, Q, Q_final, R, R_delta, estimated_disturbances_, kStateSize, 14, kControlHorizonSteps, MV_scale, OV_scale);

	Eigen::Matrix<double, (4), 1>       roll_last_control_signal, pitch_last_control_signal;

	roll_last_control_signal << 0.0, 0.0, 0.0, 0.0;
	pitch_last_control_signal << 0.0, 0.0, 0.0, 0.0;

	target_input_pitch.setZero();
	target_input_roll.setZero();

	target_state_pitch.setZero();
	target_state_roll.setZero();

	current_state_pitch.setZero();
	current_state_roll.setZero();

	estimated_disturbances_.setZero();

	target_state_roll(6, 0) = 18 * (3.141592654 / 180);
	target_state_pitch(6, 0) = 30 * (3.141592654 / 180);


	roll_cost.set_disturbance(estimated_disturbances_);
	roll_cost.set_x_ss(target_state_roll);
	roll_cost.set_u_ss(target_input_roll);
	roll_cost.set_u_current(roll_last_control_signal);
	roll_cost.set_x0_(current_state_roll);


	pitch_cost.set_disturbance(estimated_disturbances_);
	pitch_cost.set_x_ss(target_state_pitch);
	pitch_cost.set_u_ss(target_input_pitch);
	pitch_cost.set_u_current(pitch_last_control_signal);
	pitch_cost.set_x0_(current_state_pitch);


	Matrix<double, 2 * kControlHorizonSteps, 1> roll_signal_x, pitch_signal_x;
	roll_signal_x.setZero();
	pitch_signal_x.setZero();

	Matrix<double, kInputSize, 1> u_roll, u_pitch;
	u_roll.setZero();
	u_pitch.setZero();

	std::ofstream myfile_roll, myfile_pitch;
	myfile_roll.open("current_state_roll.csv");
	myfile_pitch.open("current_state_pitch.csv");

	auto t1 = Clock::now();
	auto t2 = Clock::now();

	ctpl::thread_pool p(2 /* two threads in the pool */);
	std::vector<std::future<void>> results(2);

	for (int iter = 0; iter < 1000; iter++) {

		if ((iter % 300) == 0 && iter > 0) {
			target_state_roll(6, 0) = (-1 * target_state_roll(6, 0));
			roll_cost.set_x_ss(target_state_roll);

			target_state_pitch(6, 0) = (-1 * target_state_pitch(6, 0));
			pitch_cost.set_x_ss(target_state_pitch);
		}



		

		t1 = Clock::now();

		results[0] = p.push([&roll_cost, &roll_signal_x](int id) {
			roll_signal_x = roll_cost.Evaluate(roll_signal_x); });
		
		results[1] = p.push([&pitch_cost, &pitch_signal_x](int id) {
			pitch_signal_x = pitch_cost.Evaluate(pitch_signal_x); });
		
		for (int j = 0; j < 2; ++j) {
			results[j].get();
		}


		t2 = Clock::now();

		std::cout << "Delta t2-t1: "
			<< std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
			<< " nanoseconds" << std::endl;
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
			<< " milliseconds" << std::endl;
		u_roll << roll_signal_x(0 * kControlHorizonSteps, 0), roll_signal_x(0 * kControlHorizonSteps, 0),
								roll_signal_x(1 * kControlHorizonSteps, 0), -roll_signal_x(1 * kControlHorizonSteps, 0);

		u_pitch << pitch_signal_x(0 * kControlHorizonSteps, 0), pitch_signal_x(0 * kControlHorizonSteps, 0),
			pitch_signal_x(1 * kControlHorizonSteps, 0), -pitch_signal_x(1 * kControlHorizonSteps, 0);

		current_state_roll = (model_A_70_ms * current_state_roll + model_B_70_ms * u_roll).eval();
		roll_cost.set_x0_(current_state_roll);
		roll_cost.set_u_current(u_roll);


		current_state_pitch = (model_A_70_ms * current_state_pitch + model_B_70_ms * u_pitch).eval();
		pitch_cost.set_x0_(current_state_pitch);
		pitch_cost.set_u_current(u_pitch);

		for (int print_iter = 0; print_iter < kStateSize; print_iter++) {
			myfile_roll << current_state_roll(print_iter, 0) << ",";
		}
		
		myfile_roll << "\n";

		for (int print_iter = 0; print_iter < kStateSize; print_iter++) {
			myfile_pitch << current_state_pitch(print_iter, 0) << ",";
		}

		myfile_pitch << "\n";
		

	}
	p.stop(false);
	myfile_pitch.close();
	myfile_roll.close();

	std::cout << "Gotovo" << std::endl;
	std::cin.get();
}
