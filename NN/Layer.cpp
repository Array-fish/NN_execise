#include<random>
#include<iostream>
#include"Layer.h"
layer::layer(int num_rows, int num_inputs, double epsilon)// いろいろ初期化しているだけ
	: num_rows(num_rows)
	, num_inputs(num_inputs)
	, epsilon(epsilon)
{
	weights = vector<vector<double>>(num_rows, vector<double>(num_inputs + 1, 0));
	sum_errors_for_patch = vector<vector<double>>(num_rows, vector<double>(num_inputs + 1, 0));
	outputs = vector<double>(num_rows);
	init_weight();
	dL_dx = vector<double>(num_rows);
}

void layer::init_weight() {
	std::random_device rnd;     // 非決定的な乱数生成器を生成
	std::mt19937 mt(rnd());     //  
	std::uniform_real_distribution<> rand01(0, 1.0);    // [0, 1.0] 範囲の一様乱数
	for (int row = 0; row < num_rows; ++row) {
		for (int input = 0; input < num_inputs + 1; ++input) {
			weights[row][input] = rand01(mt);
		}
	}
}

void layer::calc_outputs() {
	for (int row= 0; row < num_rows; ++row) {
		double sum_out=0;
		for (int input = 0; input < num_inputs + 1; ++input) {
			sum_out += inputs[input] * weights[row][input];
		}
		outputs[row] = sigmoid(sum_out);
	}
}

void layer::calc_dL_dx_for_before() {
	vector<double> tmp_dL_dx(num_inputs);
	for (int input = 0; input < num_inputs; ++input) {//バイアスを除いたインプットの分だけやる
		double sum = 0;
		for (int row = 0; row < num_rows; ++row) {
			// weights[row][0]はバイアスに対する重みなのでdL_dxを計算する必要がない
			// よってinput + 1で0の部分を省いている
			sum += dL_dx[row] * outputs[row] * (1 - outputs[row]) * weights[row][input + 1];
		}
		tmp_dL_dx[input] = sum;
	}
	dL_dx_for_before = tmp_dL_dx;
}

void layer::update_weights() {
	for (int row = 0; row < num_rows; ++row) {
		for (int input = 0; input < num_inputs + 1; ++input) {
			weights[row][input] -= epsilon * (1 - outputs[row]) * outputs[row] * inputs[input] * dL_dx[row];
		}
	}
}

void layer::pool_errors() {
	for (int row = 0; row < num_rows; ++row) {
		for (int input = 0; input < num_inputs + 1; ++input) {
			sum_errors_for_patch[row][input] += epsilon * (1 - outputs[row]) * outputs[row] * inputs[input] * dL_dx[row];
		}
	}
}

void layer::update_weights_for_patch(int data_size) {
	for (int row = 0; row < num_rows; ++row) {
		for (int input = 0; input < num_inputs + 1; ++input) {
			weights[row][input] -= sum_errors_for_patch[row][input]/data_size;
		}
	}
}

void layer::print_weight() {
	for (int row=0; row < num_rows; ++row) {
		for (int input = 0; input < num_inputs + 1; ++input) {
			cout << "row:" << row << ", input:" << input << ", weight:" << weights[row][input] << endl;
		}
	}
}
