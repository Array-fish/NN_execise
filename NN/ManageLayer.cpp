#include<cmath>
#include<iostream>
#include<numeric>
#include "Layer.h"
#include"ManageLayer.h"
ManageLayer::ManageLayer(int num_layer,int num_rows,int num_input,int num_output,double epsilon)
	: num_layer(num_layer)
	, num_rows(num_rows)
	, num_input(num_input)
	, num_output(num_output)
	, epsilon(epsilon)
{
	output_layer = new layer(num_output, num_rows, epsilon);
	for (int l = 0; l < num_layer; ++l) {
		// �ŏ��̒��ԑw�̓��͂̐��́C���̂܂܃f�[�^�̓��͂̐�
		// ����ȍ~�̒��ԑw�̓��͂̐��͒��ԑw�̑f�q�̐�(layer�N���X�̈����̓��͂̐��̓o�C�A�X���܂܂Ȃ�����)
		if (l == 0) {
			middle_layers.push_back(layer(num_rows, num_input, epsilon));
		}
		else {
			middle_layers.push_back(layer(num_rows, num_rows, epsilon));
		}
	}
}
vector<double> ManageLayer::forword(const vector<double>& input) {
	for (int l = 0; l < num_layer; ++l) {
		// �ŏ��̒��ԑw�������͂�����������炤
		// ����ȍ~�͈�O�̒��ԑw�̏o�͂���͂Ƃ��Ă�����Ă���
		if (l == 0) {
			middle_layers[l].set_inputs(input);
		}
		else {
			middle_layers[l].set_inputs(middle_layers[l - 1].get_outputs());
		}
		middle_layers[l].comp_outputs();
	}
	// �o�͑w�͍Ō�̒��ԑw�̏o�͂���͂Ƃ��Ă�����Ă���
	output_layer->set_inputs(middle_layers[middle_layers.size() - 1].get_outputs());
	output_layer->comp_outputs();
	// �o�͑w�̏o�͂�Ԃ�l�ŕԂ��Ă���
	return output_layer->get_outputs();
}
void ManageLayer::back_online(vector<double> &error) {
	// �^�l�Əo�͑w�̏o�͂̍����덷�Ƃ��ďo�͑w�̍X�V�pdL_dx�ɃZ�b�g����
	output_layer->set_dL_dx(error);
	// �X�V�pdL_dx���g���đO�̑w�ɓn��dL_dx���Z�o����@�d�ݍX�V�̑O�ɂ�邱��
	output_layer->comp_dL_dY();
	// �X�V�pdL_dx���g���ďd�݂��X�V
	output_layer->update_weights();
	//�ȉ����l�� ���ׂĂ̒��ԑw�̏d�݂��X�V
	for (int l = num_layer - 1; l >= 0; --l) {
		if (l == num_layer - 1) {
			middle_layers[l].set_dL_dx(output_layer->get_dL_dY_for_before());
		}
		else {
			middle_layers[l].set_dL_dx(middle_layers[l + 1].get_dL_dY_for_before());
		}
		middle_layers[l].comp_dL_dY();
		middle_layers[l].update_weights();
	}
}
// �قڏ��back_online�Ɠ���
// ������������͏d�݂̍X�V�����Ȃ��ŁC�덷�����߂�
void ManageLayer::pool_errors_patch(const vector<double>& error) {
	output_layer->set_dL_dx(error);
	output_layer->comp_dL_dY();
	output_layer->pool_errors();
	for (int l = num_layer - 1; l >= 0; --l) {
		if (l == num_layer - 1) {
			middle_layers[l].set_dL_dx(output_layer->get_dL_dY_for_before());
		}
		else {
			middle_layers[l].set_dL_dx(middle_layers[l + 1].get_dL_dY_for_before());
		}
		middle_layers[l].comp_dL_dY();
		middle_layers[l].pool_errors();
	}
}
//�S�Ă̑w�ł��߂��덷���g���ďd�݂��X�V����
void ManageLayer::back_patch(int data_size) {
	output_layer->update_weights_for_patch(data_size);
	for (int l = num_layer - 1; l >= 0; --l) {
		middle_layers[l].update_weights_for_patch(data_size);
	}
}

void ManageLayer::online(const vector<vector<double>> &input_data, vector<vector<double>> &output_data) {
	// �{���͌덷�Ń��[�v�̏I�����R���g���[�����Ȃ���΂Ȃ�Ȃ����ʓ|�Ȃ̂ň��񐔂ŏI���悤�ɂ���
	double ave_gosa =100;
	for (int times = 0; times < 10001 && ave_gosa / input_data.size() > 0.01; ++times) {
		ave_gosa = 0;
		for (int d = 0; d < input_data.size(); d++) {
			// �����ŏ������Ƌt�`��������Ă���
			vector<double> error(output_data[0].size());
			vector<double> result = forword(input_data[d]);
			for (int out = 0; out < output_data[0].size(); ++out) {
				error[out] = result[out] - output_data[d][out];
			}
			back_online(error);
			// �덷�̎Z�o�@����͂����Ɠ����Ă邩���ׂ���̂ŕʂɍX�V�Ɏg���Ă�킯����Ȃ��D
			double gosa = 0;
			for (int i = 0; i < error.size(); i++) {
				gosa += pow(error[i], 2);
			}
			ave_gosa += gosa;
			/*if (times % 100 == 0) {
				cout << "times:" << times << ", " << "gosa:" << gosa << endl;
			}*/
		}
		if (times % 10 == 0) {
			cout << "times:" << times << ", ave_gosa:" << ave_gosa / input_data.size() << endl;
		}	
	}
	/*while(true) {
		cout << "input(push Enter each number):";
		vector<double> tmp_input;
		for (int i = 0; i < input_data[0].size(); ++i) {
			double ttmp;
			cin >> ttmp;
			tmp_input.push_back(ttmp);
		}
		vector<double> ans = forword(tmp_input);
		cout << "ans:[";
		for (int i = 0; i < ans.size(); ++i) {
			cout << ans[i] << " ";
		}
		cout << "]" << endl;
		cout << "continue?(Y/n):";
		string str;
		cin >> str;
		if (str == "n")
			break;
	}*/
}
// ���online�Ɠ���
void ManageLayer::patch(const vector<vector<double>>& input_data, vector<vector<double>>& output_data) {
	double ave_gosa=100;
	for (int times = 0; times < 100001 && ave_gosa / input_data.size() > 0.01; ++times) {
		ave_gosa = 0;
		for (int d = 0; d < input_data.size(); d++) {
			vector<double> error(output_data[0].size());
			vector<double> result = forword(input_data[d]);
			for (int out = 0; out < output_data[0].size(); ++out) {
				error[out] = result[out] - output_data[d][out];
			}
			pool_errors_patch(error);//
			double gosa = 0;
			for (int i = 0; i < error.size(); i++) {
				gosa += pow(error[i], 2);
			}
			ave_gosa += gosa;
		}
		// �����ł��߂��덷���g���Ĉ�C�ɋt�`�d�������Ă���
		back_patch(input_data.size());
		if (times % 10 == 0) {
			cout << "times:" << times << ", ave_gosa:" << ave_gosa / input_data.size() << endl;
		}

	}
	/*while (true) {
		cout << "input(push Enter each number):";
		vector<double> tmp_input;
		for (int i = 0; i < input_data[0].size(); ++i) {
			double ttmp;
			cin >> ttmp;
			tmp_input.push_back(ttmp);
		}
		vector<double> ans = forword(tmp_input);
		cout << "ans:[";
		for (int i = 0; i < ans.size(); ++i) {
			cout << ans[i] << " ";
		}
		cout << "]" << endl;
		cout << "continue?(Y/n):";
		string str;
		cin >> str;
		if (str == "n")
			break;
	}*/
}
void ManageLayer::test(const vector<vector<double>>& test_input_data, vector<vector<double>>& test_output_data) {
	double ave_gosa = 0;
	for (int d = 0; d < test_input_data.size(); d++) {
		vector<double> error(test_output_data[0].size());
		vector<double> result = forword(test_input_data[d]);
		for (int out = 0; out < test_output_data[0].size(); ++out) {
			error[out] = result[out] - test_output_data[d][out];
		}
		double gosa = 0;
		for (int i = 0; i < error.size(); i++) {
			gosa += pow(error[i], 2);
		}
		ave_gosa += gosa;
	}
	cout << "test_ave_gosa:" << ave_gosa / test_input_data.size() << endl;
}
void ManageLayer::print_weight() {
	for (int l = 0; l < middle_layers.size(); ++l) {
		middle_layers[l].print_weight();
	}
	output_layer->print_weight();
}