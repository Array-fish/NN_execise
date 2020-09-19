#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include"ManageLayer.h"
#include"Layer.h"
using namespace std;
vector<vector<double>> get_vector_from_file(string filename);
int main() {
	// ユーザ入力部
	int soshi,sou,is_online;
	cout << "素子数:";
	cin >> soshi;
	cout << "層数:";
	cin >> sou;
	cout << "一括学習:0, 逐次学習:1" << endl <<":";
	cin >> is_online;
	// データ読み取り部
	vector<vector<double>> input_data = get_vector_from_file("data.csv");
	vector<vector<double>> output_data = get_vector_from_file("data_T.csv");
	vector<vector<double>> test_input_data = get_vector_from_file("dis_sig.csv");
	vector<vector<double>> test_output_data = get_vector_from_file("dis_T_sig.csv");
	// ここからNN本体
	// ManageLayer クラスの第4引数が学習率だから変えたければいじって
	ManageLayer ml(sou, soshi, input_data[0].size(), output_data[0].size() , 0.5);
	if (is_online) {
		ml.online(input_data,output_data);
	}
	else {
		ml.patch(input_data,output_data);
	}
	//ml.test(test_input_data,test_output_data);
	return 0;
}
vector<vector<double>> get_vector_from_file(string filename) {
	ifstream ifs(filename);
	if (ifs.fail()) {
		std::cerr << "失敗" << std::endl;
	}
	string str, str1;
	vector<vector<double>> data;
	while (getline(ifs, str)) {
		stringstream ss{ str };
		vector<double> tmp;
		while (getline(ss, str1, ',')) {
			tmp.push_back(stod(str1));
		}
		data.push_back(tmp);
	}
	return data;
}