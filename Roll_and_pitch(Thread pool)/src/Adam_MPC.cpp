#include "Adam_MPC.h"



Adam_MPC::Adam_MPC(Matrix<double, num_state_variables, num_state_variables> A, Matrix<double, num_state_variables, num_manipulated_variables>  B,
	Matrix<double, num_state_variables, num_state_variables> Bd, Matrix<double, num_state_variables, num_state_variables> Q,
	Matrix<double, num_state_variables, num_state_variables> Q_final, Matrix<double, num_manipulated_variables, num_manipulated_variables> R,
	Matrix<double, num_manipulated_variables, num_manipulated_variables> R_delta,
	Matrix<double, num_state_variables, 1> disturbance, int num_params, int pred_horizon, int control_horizon,
	Matrix<double, num_manipulated_variables, num_manipulated_variables> scale_MV, Matrix<double, num_state_variables, num_state_variables> scale_OV) : num_params_(num_params), pred_horizon(pred_horizon), control_horizon(control_horizon), A_(A), B_(B), Bd_(Bd), Q_(Q), Q_final_(Q_final), R_(R), R_delta_(R_delta), disturbance_(disturbance)
{
	//this->insecure_ = this->Bd_ * disturbance;
	x_states.setZero();
	deriv_wrt_u.setZero();
	u.setZero();
	u_past.setZero();
	u_current.setZero();
	u_ss_.setZero();
	lambdas_x.setZero();
	lambdas_u.setZero();
	lambdas_u_ref.setZero();
	lambdas_u_ref.setZero();
	x0_.setZero();
	change_x.setZero();
	x_ss_.setZero();
	gradients.setZero();
	gradient_past.setZero();
	gradient_square_past.setZero();
	gradient_past_tilda.setZero();
	gradient_square_past_tilda.setZero();
	B_x_u.setZero();
	epsilon = 0.04;
	rho1 = 0.9;
	rho2 = 0.999;
	delta = 1e-7;
	residuum = 0.0;
	max_iter = 5;
	alfa = 200;
	saturation_count = 0;
	min_residuum = 1e-5;
	count_jacobians = 0;
	scale_MV_inv = scale_MV.inverse();
	scale_OV_inv = scale_OV.inverse();
	A_pow_B_cache.setZero();
	A_pow_B_cache.block(0, 0, A_.rows(), B_.cols()) = MatrixXd::Identity(A_.rows(), A_.cols())* B_;

	for (int i = 0; i < pred_horizon - 1; i++) {

		A_pow_B_cache.block(0, (i + 1)*B_.cols(), A_.rows(), B_.cols()) = (A_* A_pow_B_cache.block(0, (i)*B_.cols(), A_.rows(), B_.cols())).eval();

	}

	du_limit(0, 0) = 70 * 1e-3 * 2;
	du_limit(0, 1) = 2;
	du_limit(1, 0) = -70 * 1e-3 * 2;
	du_limit(1, 1) = -2;
	u_limit(0, 0) = 0.6 / 2 - 0.01;
	u_limit(0, 1) = 50;
	u_limit(1, 0) = -u_limit(0, 0);
	u_limit(1, 1) = -u_limit(0, 1);



}

//When you add the const keyword to a method the this pointer will essentially become a pointer to const object, and you cannot therefore change any member data. (This is not completely true, because you can mark a member as mutable and a const method can then change it. It's mostly used for internal counters and stuff.).

Matrix<double, 2 * mpc_control_horizon, 1>& Adam_MPC::Evaluate(Matrix<double, 2 * mpc_control_horizon, 1>&   x) {

	saturation_count = 0;
	if (new_desire_state) {
		new_desire_state = false;
		residuum_old = 1000.0;
		gradient_past.setZero();
		gradient_square_past.setZero();
		t = 0.0;
	}
	x_states.block(0, 0, x0_.rows(), x0_.cols()) = x0_;
	/*std::ofstream myfile;
	myfile.open(string("jacobians")+ std::to_string(count_jacobians) + string(".csv"));
	count_jacobians++;*/
	for (int iter = 0; iter < max_iter; iter++) {
		///TODO: Ove tri linije izbacit
		//deriv_wrt_u.setZero();
		//x_states.setZero();


		u_past = 1 * u_current;



		for (int i = 0; i < pred_horizon; i++) {


			/*if (i < control_horizon) {
				u << x(0 * control_horizon + (i), 0), x(0 * control_horizon + (i), 0),
					x(1 * control_horizon + (i), 0), -x(1 * control_horizon + (i), 0);

			}*/

			switch (i) {
			case 0:
			case 1:
			case 2:
			case 3:
			case 4:
				u << x(0 * control_horizon + (i), 0), x(0 * control_horizon + (i), 0),
					x(1 * control_horizon + (i), 0), -x(1 * control_horizon + (i), 0);
				B_x_u = B_ * u;
				break;
			default:
				break;
			}



			x_states.block(0, i + 1, x0_.rows(), x0_.cols()) = (A_ * x_states.block(0, i, x0_.rows(), x0_.cols()) + B_x_u).eval();
			lambdas_x.block(0, i, x0_.rows(), x0_.cols()) = -1 * x_ss_ + x_states.block(0, i, x0_.rows(), x0_.cols());



			lambdas_u.block(0, i, u_past.rows(), u_past.cols()) = u - u_past;
			lambdas_u_ref.block(0, i, u.rows(), u.cols()) = u - u_ss_;


			//derivation of u
			//if (i < control_horizon) {
			//	//deriv_wrt_u.block(0, i, u.rows(), u.cols()) = (deriv_wrt_u.block(0, i, u.rows(), u.cols()) + (2 * R_*u) - 2 * R_*u_ss_ + (4 * R_delta_*u) + (-2 * R_delta_*u_past));
			//	deriv_wrt_u.block(0, i, u.rows(), u.cols()) = ((2 * R_*u) - 2 * R_*u_ss_ + (4 * R_delta_*u) + (-2 * R_delta_*u_past));


			//	if (i > 0) {
			//		deriv_wrt_u.block(0, i - 1, u.rows(), u.cols()) = (deriv_wrt_u.block(0, i - 1, u.rows(), u.cols()) - 2 * R_delta_*u);

			//	}
			//}
			//else {
			//	deriv_wrt_u.block(0, control_horizon - 1, u.rows(), u.cols()) = (deriv_wrt_u.block(0, control_horizon - 1, u.rows(), u.cols()) + (2 * R_*u) - 2 * R_*u_ss_ + (4 * R_delta_*u) + (-2 * R_delta_*u_past));//.eval();


			//	deriv_wrt_u.block(0, control_horizon - 1, u.rows(), u.cols()) = (deriv_wrt_u.block(0, control_horizon - 1, u.rows(), u.cols()) - 2 * R_delta_*u);


			//}

			switch (i) {
			case 0:
				deriv_wrt_u.block(0, i, u.rows(), u.cols()) = ((2 * R_*u) - 2 * R_*u_ss_ + (4 * R_delta_*u) + (-2 * R_delta_*u_past));
				break;
			case 1: case 2: case 3: case 4:
				deriv_wrt_u.block(0, i, u.rows(), u.cols()) = ((2 * R_*u) - 2 * R_*u_ss_ + (4 * R_delta_*u) + (-2 * R_delta_*u_past));
				deriv_wrt_u.block(0, i - 1, u.rows(), u.cols()) = (deriv_wrt_u.block(0, i - 1, u.rows(), u.cols()) - 2 * R_delta_*u).eval();
				break;
			default:
				deriv_wrt_u.block(0, control_horizon - 1, u.rows(), u.cols()) = (deriv_wrt_u.block(0, control_horizon - 1, u.rows(), u.cols()) + (2 * R_*u) - 2 * R_*u_ss_ + (4 * R_delta_*u) + (-2 * R_delta_*u_past)).eval();


				deriv_wrt_u.block(0, control_horizon - 1, u.rows(), u.cols()) = (deriv_wrt_u.block(0, control_horizon - 1, u.rows(), u.cols()) - 2 * R_delta_*u).eval();
				break;

			}



			//derivation of x
			for (int j = 0; j <= i; j++) {


				//if (j < control_horizon && i < control_horizon) {

				//	deriv_wrt_u.block(0, j, u.rows(), u.cols()) = (deriv_wrt_u.block(0, j, u.rows(), u.cols()) + ((2 * Q_*x_states.block(0, i + 1, x0_.rows(), x0_.cols()) - 2 * Q_*x_ss_).transpose()*A_pow_B_cache.block(0, (i - j)*B_.cols(), A_.rows(), B_.cols())).transpose());//.eval();

				//}
				//else {
				//	if (j >= control_horizon) {
				//		deriv_wrt_u.block(0, 4, u.rows(), u.cols()) = (deriv_wrt_u.block(0, 4, u.rows(), u.cols()) + ((2 * Q_*x_states.block(0, i + 1, x0_.rows(), x0_.cols()) - 2 * Q_*x_ss_).transpose()*A_pow_B_cache.block(0, (i - j)*B_.cols(), A_.rows(), B_.cols())).transpose());//.eval();

				//	}
				//	else {
				//		deriv_wrt_u.block(0, j, u.rows(), u.cols()) = (deriv_wrt_u.block(0, j, u.rows(), u.cols()) + ((2 * Q_*x_states.block(0, i + 1, x0_.rows(), x0_.cols()) - 2 * Q_*x_ss_).transpose()*A_pow_B_cache.block(0, (i - j)*B_.cols(), A_.rows(), B_.cols())).transpose());//.eval();

				//	}

				//}


				switch (j)
				{
				case 0: case 1: case 2: case 3: case 4:
					deriv_wrt_u.block(0, j, u.rows(), u.cols()) = (deriv_wrt_u.block(0, j, u.rows(), u.cols()) + ((2 * Q_*x_states.block(0, i + 1, x0_.rows(), x0_.cols()) - 2 * Q_*x_ss_).transpose()*A_pow_B_cache.block(0, (i - j)*B_.cols(), A_.rows(), B_.cols())).transpose()).eval();
					break;

				default:
					deriv_wrt_u.block(0, 4, u.rows(), u.cols()) = (deriv_wrt_u.block(0, 4, u.rows(), u.cols()) + ((2 * Q_*x_states.block(0, i + 1, x0_.rows(), x0_.cols()) - 2 * Q_*x_ss_).transpose()*A_pow_B_cache.block(0, (i - j)*B_.cols(), A_.rows(), B_.cols())).transpose()).eval();
					break;
				}


			}

			u_past = u;
		}

		lambdas_u_ref = scale_MV_inv * lambdas_u_ref;
		lambdas_u = scale_MV_inv * lambdas_u;
		lambdas_x = scale_OV_inv * lambdas_x;




		residuum_signal = (lambdas_u_ref.cwiseProduct(R_*lambdas_u_ref)).sum() + (lambdas_u.cwiseProduct(R_delta_*lambdas_u)).sum();

		residuum_state = (lambdas_x.cwiseProduct(Q_*lambdas_x)).sum();

		residuum = residuum_signal + residuum_state;





		for (int iter_der = 0; iter_der < control_horizon; iter_der++) {
			//derivation df/du1
			gradients(iter_der, 0) = 2 * deriv_wrt_u(0, iter_der);
			//derivation df/du2
			gradients(iter_der + mpc_control_horizon, 0) = 2 * deriv_wrt_u(2, iter_der);
		}

		gradient_square_past = (rho2*gradient_square_past + (1 - rho2)*gradients.cwiseProduct(gradients)).eval();

		gradient_past = (rho1*gradient_past + (1 - rho1)*gradients).eval();

		/*cout << "gradienti" << endl << gradient_past << endl << endl;
		cout << "gradient_square_past" << endl << gradient_square_past << endl << endl;*/

		t += 1.0;

		for (int adam_iter = 0; adam_iter < num_heuristic_variables*mpc_control_horizon; adam_iter++) {
			gradient_past_tilda(adam_iter, 0) = (gradient_past(adam_iter, 0)) / (1.0 - pow(rho1, t));
			gradient_square_past_tilda(adam_iter, 0) = (gradient_square_past(adam_iter, 0)) / (1.0 - pow(rho2, t));
		}

		/*cout << "gradient_past_tilda" << endl << gradient_past_tilda << endl << endl;
		cout << "gradient_square_past_tilda" << endl << gradient_square_past_tilda << endl << endl;*/

		/*if (residuum < min_residuum) {
			return x;
		}*/


		/*
		for (int print_iter = 0; print_iter < 2*mpc_control_horizon; print_iter++) {
			myfile << Jacobian( 0, print_iter) << ",";
		}
		myfile << "\n";*/


		//cout << "Jacobians" <<endl << Jacobian << endl;
		//Radi !!


		for (int adam_iter = 0; adam_iter < num_heuristic_variables*mpc_control_horizon; adam_iter++) {
			change_x(adam_iter, 0) = -1 * (epsilon * gradient_past_tilda(adam_iter, 0)) / (sqrt(gradient_square_past_tilda(adam_iter, 0)) + delta);
		}
		//cout << "change_x" << endl << change_x << endl << endl;
		x = (x + change_x).eval();



		/*if ((residuum - residuum_old) > 1e-4) {
			x = this->check_bounderies(x);
			return x;
		}*/


		residuum_old = residuum;

	}

	//myfile.close();
	x = this->check_bounderies(x);
	return x;
}


Adam_MPC::~Adam_MPC()
{
}

Matrix<double, 2 * mpc_control_horizon, 1>& Adam_MPC::check_bounderies(Matrix<double, 2 * mpc_control_horizon, 1>&  x) {
	dummy_u = u_current;
	for (int i = 0; i < mpc_control_horizon; i++) {

		for (int j = 0; j < num_heuristic_variables; j++) {

			if (x(j * mpc_control_horizon + (i), 0) > dummy_u(2 * j, 0) + du_limit(0, j)) {
				/*cout << "dummy u " << endl << dummy_u << endl;
				cout << "du_lim" << endl << du_limit << endl;
				cout << x(j * mpc_control_horizon + (i), 0) << "   " << dummy_u(2 * j, 0) << "  " << du_limit(0, j)  << "  " << u_limit(0, j)<< endl;*/


				x(j * mpc_control_horizon + (i), 0) = dummy_u(2 * j, 0) + du_limit(0, j);




			}
			else if (x(j * mpc_control_horizon + (i), 0) < dummy_u(2 * j, 0) + du_limit(1, j)) {
				/*cout << "dummy u " << endl << dummy_u << endl;
				cout << "du_lim" << endl << du_limit << endl;
				cout << x(j * mpc_control_horizon + (i), 0) << "   " << dummy_u(2 * j, 0) << "  " << du_limit(1, j) << "  " << u_limit(1, j) << endl;*/

				x(j * mpc_control_horizon + (i), 0) = dummy_u(2 * j, 0) + du_limit(1, j);


			}

			if (x(j * mpc_control_horizon + (i), 0) > u_limit(0, j)) {
				x(j * mpc_control_horizon + (i), 0) = u_limit(0, j);
			}
			else if (x(j * mpc_control_horizon + (i), 0) < u_limit(1, j)) {
				x(j * mpc_control_horizon + (i), 0) = u_limit(1, j);
			}


		}
		//cout << "x " << endl << x << endl;
		dummy_u << x(0 * mpc_control_horizon + (i), 0), x(0 * mpc_control_horizon + (i), 0),
			x(1 * mpc_control_horizon + (i), 0), -x(1 * mpc_control_horizon + (i), 0);
	}
	return x;


}
