package knn;
import java.util.*;
import java.io.*;

public class KNN_main {
	
public static void main(String[] args) throws NumberFormatException, IOException {
		
		System.out.println("Welcome, This program is implementation of KNN algorthim for Women Diabetes Dataset");
		Scanner sc=new Scanner(System.in);
		KNN_Implementation trn_ds=new KNN_Implementation();
		String trainfilename="/home/namanb99/eclipse-workspace/Minor_2/src/knn/diabetesTrainData.csv";
		String testfilename="/home/namanb99/eclipse-workspace/Minor_2/src/knn/diabetesTestData.csv";
		
		trn_ds.getKValueandDistMetrics();
		trn_ds.loadtestData(testfilename);
		trn_ds.loadtrainLabelData(trainfilename);
		  trn_ds.distanceCalculate();
		 
		 
		  sc.close();

	}


}
