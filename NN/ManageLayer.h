#ifndef MANAGELAYER_H
#define MANAGELAYER_H
#include<vector>
using namespace std;
class layer;
class ManageLayer {
private:
	// var
	const int num_layer;// ���C���[�̐�
	const int num_rows;// �f�q�̐�
	const int num_input;// ���͂̐�
	const int num_output;// �o�͂̐�
	const double epsilon;// �w�K��
	layer* output_layer;// �o�͑w
	vector<layer> middle_layers;// ���ԑw
	// func
	vector<double> forword(const vector<double>& input);// ������
	void back_online(vector<double> &error);// online�̋t����
	void back_patch(int data_size);// patch�̋t�����ŏd�݂̍X�V��������
	void pool_errors_patch(const vector<double> &error);// patch�̋t�����Ō덷��~�ς�����
public:
	ManageLayer(int num_layer,int num_rows,int num_input,int num_output,double epsilon);
	virtual ~ManageLayer() {};
	void online(const vector<vector<double>> &input_data, vector<vector<double>> &output_data);
	void patch(const vector<vector<double>>& input_data, vector<vector<double>>& output_data);
	void test(const vector<vector<double>>& test_input_data, vector<vector<double>>& test_output_data);
	void print_weight();// �f�o�b�N�p�̊֐�
};
#endif MANAGELAYER_H
