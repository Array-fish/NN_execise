#ifndef LAYER_H
#define LAYER_H
#include<vector>
using namespace std;
class layer {
private:
	// var
	const double epsilon;                        // 学習率
	const int num_rows;                          // パーセプトロンの数
	vector<vector<double>> weights;              // 重み[ぱーせぷトロンの数][入力+1]　+1はバイアス分ね
	vector<double> inputs;                       // 入力　ここにはバイアスが含まれる
	const int num_inputs;                        // 入力の数 バイアスは含まない
	vector<double> outputs;                      // 出力　数はパーセプトロンの数と同じ
	vector<double> dL_dx;                        // この層が更新に使うための後の層から持ってきたやつ
	vector<double> dL_dx_for_before;             // 前の層が更新に使うやつ 
	vector<vector<double>> sum_errors_for_patch; // パッチ学習のときにデータ全部の誤差をためとくやつ
	// func
	void init_weight();                          // 乱数で初期化[0, 1]
	double sigmoid(double x, double gain = 1);
public:

	layer(int num_rows, int num_inputs, double epsilon); // 引数はパーセプトロンの数と入力の数，学習率
	virtual ~layer() {};
	void set_inputs(const vector<double>& inputs);       // 入力を受け取る用
	void calc_outputs();                                 // 出力を計算する
	vector<double> get_outputs();                        // 出力結果を取るやつ 後の層が使う
	void set_dL_dx(const vector<double>& dL_dx);         // 後ろからとってきた誤差を取り込む用
	void calc_dL_dx_for_before();                                   // 前の層が使う誤差dL/dYを作る　重み更新の前に使ってね
	void update_weights();                               // バイアスに対するものを含む重みの更新　
	vector<double> get_dL_dx_for_before();               // 前の層が使う誤差dL/dYを渡すとき用
	void pool_errors();                                  // 誤差を蓄積する
	void update_weights_for_patch(int data_size);        // パッチ学習でsum_errors_for_patchを使って重みを更新する
	void reset_weights_variation();                      // 蓄積された誤差をリセットする。
	void print_weight();                                 // デバック用関数
};

inline double layer::sigmoid(double x, double gain)
{
	return 1.0 / (1.0 + exp(-gain * x));
}
inline void layer::set_inputs(const vector<double>& inputs) {
	this->inputs = inputs;
	this->inputs.insert(this->inputs.begin(), 1);// バイアス分を最初に追加
}
inline vector<double> layer::get_outputs() {
	return outputs;
}
// 後ろからとってきたのを突っ込むよう
inline void layer::set_dL_dx(const vector<double>& dL_dx) {
	this->dL_dx = dL_dx;
}
inline vector<double> layer::get_dL_dx_for_before() {
	return dL_dx_for_before;
}
inline void layer::reset_weights_variation() {
	sum_errors_for_patch = vector<vector<double>>(num_rows, vector<double>(num_inputs + 1, 0));// 初期化
}
#endif LAYER_H
