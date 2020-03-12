#include "tiny_dnn/tiny_dnn.h"
#include "CSVreader.h"

#include <fstream>
#include <vector>

using namespace std;

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

int main(){
	
	CSVreader move1_data("movement1.csv");
	CSVreader move2_data("movement2.csv");
	CSVreader move3_data("movement3.csv");
	CSVreader move4_data("movement4.csv");

	move1_data.readCSV();
	move2_data.readCSV();
	move3_data.readCSV();
	move4_data.readCSV();

	vector<vector<float>> data1 = move1_data.getVectorArray();
	vector<vector<float>> data2 = move1_data.getVectorArray();
	vector<vector<float>> data3 = move1_data.getVectorArray();
	vector<vector<float>> data4 = move1_data.getVectorArray();

    vector<vec_t> trainIn(80);
	vector<float> temp;
	
	for(int i=0; i< data1.size(); i++){
		temp = data1[i];
		for(int j=0; j<temp.size();j++){
			trainIn[i].push_back(temp[j]);
		}
		temp.clear();
	}
	for(int i=0; i< data2.size(); i++){
		temp = data2[i];
		for(int j=0; j<temp.size();j++){
			trainIn[i+20].push_back(temp[j]);
		}
		temp.clear();
	}
	for(int i=0; i< data3.size(); i++){
		temp = data3[i];
		for(int j=0; j<temp.size();j++){
			trainIn[i+40].push_back(temp[j]);
		}
		temp.clear();
	}
	for(int i=0; i< data4.size(); i++){
		temp = data4[i];
		for(int j=0; j<temp.size();j++){
			trainIn[i+60].push_back(temp[j]);
		}
		temp.clear();
	}

    vector<vec_t> trainOut;
    vec_t label1 = {0};
    vec_t label2 = {0.33};
    vec_t label3 = {0.66};
    vec_t label4 = {1.0};
    
    for(int i=0; i<data1.size(); i++){
        trainOut.push_back(label1);
    }
    for(int i=0; i<data2.size(); i++){
        trainOut.push_back(label2);
    }
    for(int i=0; i<data3.size(); i++){
        trainOut.push_back(label3);
    }
    for(int i=0; i<data4.size(); i++){
        trainOut.push_back(label4);
    }

   

    //now we have all the training data :)
    /****************************************************************************************************/

    network<sequential> net;
    /*
    *Param: number of layer inputs
    *Param: number of layer outputs
    *Uses sigmoid function -> relu() and softmax() also are available...
    */
    net << fully_connected_layer(220,64) << relu() //relu 
        << fully_connected_layer(64,64) << relu()
        << fully_connected_layer(64,1) << softmax(); //softmax


	//adam also available...
	//gradient_descent optimizer; //learning rate of 0.53
	adam optimizer;	
	optimizer.alpha = 0.01f;
	//loss function mse
	//batch size of 1, epochs == 1000
	net.fit<mse>(optimizer, trainIn, trainOut, 1, 50);


	//now we need to parse in the verification data...
	CSVreader ver_data("verificationData.csv");

	ver_data.readCSV();

	vector<vector<float>> verification = ver_data.getVectorArray();

	vector<vec_t> testIn(40);
	
	for(int i=0; i< verification.size(); i++){
		temp = verification[i];
		for(int j=0; j<temp.size();j++){
			testIn[i].push_back(temp[j]);
		}
		temp.clear();
	}

	for(int i=0; i<testIn.size();i++){
		vec_t result = net.predict(testIn[i]);
		std::cout << result[0] << endl;
					//<< result[1] << " "
					//<< result[2] << " "
					//<< result[3] << " " << endl;
	}
	
	std::cout << endl;

}

