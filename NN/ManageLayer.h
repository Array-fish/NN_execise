#ifndef MANAGELAYER_H
#define MANAGELAYER_H
#include<vector>
using namespace std;
class layer;
// 複数のLayerクラスを管理して学習と識別を行うクラス。
class ManageLayer {
private:
	// variable
	const int num_layer;         // レイヤーの数
	const int num_rows;          // 素子の数
	const int num_input;         // 入力の数
	const int num_output;        // 出力の数
	const double epsilon;        // 学習率
	vector<layer> middle_layers; // 中間層
	// function
	/** back_online, back_patch
	* 入力は教師データと出力の誤差。
	* back_onlineは逐次学習で重みを更新、back_patchは一括学習で重みを更新
	*/
	void back_online(vector<double> &error);
	void back_patch(int data_size);
	/** pool_errors_patch
	* 入力は教師データと出力の誤差。
	* back_patchで用いる誤差を蓄積するやつ
	*/
	void pool_errors_patch(const vector<double> &error);// patchの逆方向で誤差を蓄積するやつ
public:
	ManageLayer(int num_layer,int num_rows,int num_input,int num_output,double epsilon);
	virtual ~ManageLayer() {};
	/** online, patch
	* 引数は教師データの入力と出力
	* onlineは逐次学習, patchは一括学習を行う。
	*/
	void online(const vector<vector<double>> &input_data, vector<vector<double>> &output_data);
	void patch(const vector<vector<double>>& input_data, vector<vector<double>>& output_data);
	/** forword
	* 引数は入力
	* 返り値はモデルの出力
	* 入力データに対する順方向演算を行う
	*/
	vector<double> forword(const vector<double>& input);
	/** loss
	* 引数はテストデータの入力と出力
	* テストデータを入力すると識別率を算出してくれる
	*/
	void loss(const vector<vector<double>>& test_input_data, vector<vector<double>>& test_output_data);
	void print_weight();// デバック用の関数
};
#endif MANAGELAYER_H
