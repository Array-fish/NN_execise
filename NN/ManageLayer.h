#ifndef MANAGELAYER_H
#define MANAGELAYER_H
#include<vector>
using namespace std;
class layer;
class ManageLayer {
private:
	// var
	const int num_layer;// レイヤーの数
	const int num_rows;// 素子の数
	const int num_input;// 入力の数
	const int num_output;// 出力の数
	const double epsilon;// 学習率
	layer* output_layer;// 出力層
	vector<layer> middle_layers;// 中間層
	// func
	vector<double> forword(const vector<double>& input);// 順方向
	void back_online(vector<double> &error);// onlineの逆方向
	void back_patch(int data_size);// patchの逆方向で重みの更新をするやつ
	void pool_errors_patch(const vector<double> &error);// patchの逆方向で誤差を蓄積するやつ
public:
	ManageLayer(int num_layer,int num_rows,int num_input,int num_output,double epsilon);
	virtual ~ManageLayer() {};
	void online(const vector<vector<double>> &input_data, vector<vector<double>> &output_data);
	void patch(const vector<vector<double>>& input_data, vector<vector<double>>& output_data);
	void test(const vector<vector<double>>& test_input_data, vector<vector<double>>& test_output_data);
	void print_weight();// デバック用の関数
};
#endif MANAGELAYER_H
