#include<random>
#include<iostream>
#include"Layer.h"
layer::layer(int num_rows, int num_inputs, double epsilon)// ���낢�돉�������Ă��邾��
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
	std::random_device rnd;     // �񌈒�I�ȗ���������𐶐�
	std::mt19937 mt(rnd());     //  
	std::uniform_real_distribution<> rand01(0, 1.0);    // [0, 1.0] �͈͂̈�l����
	for (int row = 0; row < num_rows; ++row) {
		for (int input = 0; input < num_inputs + 1; ++input) {
			weights[row][input] = rand01(mt);
		}
	}
}

void layer::comp_outputs() {
	for (int row= 0; row < num_rows; ++row) {
		double sum_out=0;
		for (int input = 0; input < num_inputs + 1; ++input) {
			sum_out += inputs[input] * weights[row][input];
		}
		outputs[row] = sigmoid(sum_out);
	}
}

void layer::comp_dL_dY() {
	vector<double> dL_dY(num_inputs);
	for (int input = 0; input < num_inputs; ++input) {//�o�C�A�X���������C���v�b�g�̕��������
		double sum = 0;
		for (int row = 0; row < num_rows; ++row) {
			// weights[row][0]�̓o�C�A�X�ɑ΂���d�݂Ȃ̂�dL_dY���v�Z����K�v���Ȃ�
			// �����input + 1��0�̕������Ȃ��Ă���
			sum += dL_dx[row] * outputs[row] * (1 - outputs[row]) * weights[row][input + 1];
		}
		dL_dY[input] = sum;
	}
	dL_dY_for_before = dL_dY;
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
	sum_errors_for_patch = vector<vector<double>>(num_rows, vector<double>(num_inputs + 1, 0));// ������
}

void layer::print_weight() {
	for (int row=0; row < num_rows; ++row) {
		for (int input = 0; input < num_inputs + 1; ++input) {
			cout << "row:" << row << ", input:" << input << ", weight:" << weights[row][input] << endl;
		}
	}
}
