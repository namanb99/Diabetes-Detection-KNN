package knn;

import java.util.*;
import java.io.*;

public class KNN_Implementation {
	
	// created lists for storing training and testing datasets label and features.

		private List<double[]> trainfeatures = new ArrayList<>();
		private List<String> trainlabel = new ArrayList<>();

		private List<double[]> testfeatures = new ArrayList<>();
		private List<String> testlabel = new ArrayList<>();
		/*
		 * sc object for getting user input
		 */

		Scanner sc = new Scanner(System.in);
		int knn_value = 1;
		int DistanceMetricsSelection = 0;
		int totalNumberOfLabel = 0;

		/*
		 * loading testing data and extracting features and label for training dataset
		 * 
		 */
		void loadtestData(String testfilename) throws NumberFormatException, IOException {

			File testfile = new File(testfilename);

			try {
				BufferedReader testreadFile = new BufferedReader(new FileReader(testfile));
				PrintWriter pw = new PrintWriter("RealTestLabel.txt");
				String testline;
				while ((testline = testreadFile.readLine()) != null) {

					String[] split = testline.split(",");
					double[] feature = new double[split.length - 1];
					for (int i = 0; i < split.length - 1; i++)
						feature[i] = Double.parseDouble(split[i]);
					testfeatures.add(feature);
					testlabel.add(split[feature.length]);
					// writing original label for test data to file and counting number of label.
					pw.println(split[feature.length]);
					totalNumberOfLabel++;

				}
				pw.close();
				testreadFile.close();
			}

			catch (FileNotFoundException e) {
				// TODO Auto catch block
				e.printStackTrace();
			}

		}

		
		void loadtrainLabelData(String trainfilename) throws NumberFormatException, IOException {

			File trainfile = new File(trainfilename);

			try {
				BufferedReader trainreadFile = new BufferedReader(new FileReader(trainfile));
				PrintWriter pw = new PrintWriter("RealTrainLabel.txt");
				String trainline;
				while ((trainline = trainreadFile.readLine()) != null) {

					String[] split = trainline.split(",");
					double[] feature = new double[split.length - 1];
					for (int i = 0; i < split.length - 1; i++)
						feature[i] = Double.parseDouble(split[i]);
					trainfeatures.add(feature);
					trainlabel.add(split[feature.length]);
					// writing original label for test data to file and counting number of label.
					pw.println(split[feature.length]);
					totalNumberOfLabel++;

				}
				pw.close();
				trainreadFile.close();
			}

			catch (FileNotFoundException e) {
				// TODO Auto catch block
				e.printStackTrace();
			}

		}

		
		/*
		 * Based on user input, calling algorithm to calculate distance
		 */
		void distanceCalculate() throws IOException {

			if (DistanceMetricsSelection == 1) {
				euclideanTrainDistance();
				euclideanTestDistance();
				// calling accuracy method to show accuracy of model.
				accuracy();
			}

			else if (DistanceMetricsSelection == 2) {
				manhattanTrainDistance();
				manhattanTestDistance();
				accuracy();
			}

			else {
				// if user selecting invalid options then they must select correct option.
				System.out.println("Invalid Selection");
				getKValueandDistMetrics();
				distanceCalculate();
			}
		}
		
		@SuppressWarnings("unchecked")
		void euclideanTrainDistance() throws FileNotFoundException {
			KNN_Distance euclidean = new KNN_Distance();


			Iterator<double[]> trainOverfitITR = trainfeatures.iterator();

			PrintWriter pw = new PrintWriter("EuclideanTrainResult.txt");
			while (trainOverfitITR.hasNext()) {
				double trainOverfitF[] = trainOverfitITR.next();
				Iterator<double[]> trainITR = trainfeatures.iterator();
				int noOfobject = 0;
				ArrayList<DistanceAndFeatures> ts = new ArrayList<>();
				while (trainITR.hasNext()) {
					double trainF[] = trainITR.next();
					double dist = 0;
					dist = euclidean.getEuclideanDistance(trainF, trainOverfitF);
					
					String trainFeat = trainlabel.get(noOfobject);
					DistanceAndFeatures DfObject = new DistanceAndFeatures(dist, trainFeat);
					ts.add(DfObject);
					Collections.sort(ts);
					noOfobject++;

				}

				/*
				 * counting top predicted label based on k value
				 */
				int flag = 0, positive = 0, negative = 0;

				while (flag < knn_value) {
					DistanceAndFeatures s = ts.get(flag);
					String s1 = s.getLabel();
					if (s1.equals("1"))
						positive++;
					else 
						negative++;
					flag++;

				}

				/*
				 * counting label and selecting highest label count as prediction label and
				 * writing to output file.
				 */
				if (positive > negative) {
					pw.println("1");

				} else 
					pw.println("0");
			}
			pw.close();
		}

		@SuppressWarnings("unchecked")
		void euclideanTestDistance() throws FileNotFoundException {
			KNN_Distance euclidean = new KNN_Distance();

			Iterator<double[]> testITR = testfeatures.iterator();

			PrintWriter pw = new PrintWriter("EuclideanTestResult.txt");

			while (testITR.hasNext()) {
				double testF[] = testITR.next();
				Iterator<double[]> trainITR = trainfeatures.iterator();
				int noOfobject = 0;
				ArrayList<DistanceAndFeatures> ts = new ArrayList<>();
				while (trainITR.hasNext()) {
					double trainF[] = trainITR.next();
					double dist = 0;
					dist = euclidean.getEuclideanDistance(trainF, testF);

					String trainFeat = trainlabel.get(noOfobject);
					DistanceAndFeatures DfObject = new DistanceAndFeatures(dist, trainFeat);
					ts.add(DfObject);
					Collections.sort(ts);
					noOfobject++;

				}

				/*
				 * counting top predicted label based on k value
				 */
				int flag = 0, positive = 0, negative = 0;

				while (flag < knn_value) {
					DistanceAndFeatures s = ts.get(flag);
					String s1 = s.getLabel();
					if (s1.equals("1"))
						positive++;
					else 
						negative++;
					flag++;

				}

				/*
				 * counting label and selecting highest label count as prediction label and
				 * writing to output file.
				 */
				if (positive > negative) {
					pw.println("1");

				} else 
					pw.println("0");
			}
			pw.close();
		}

		
		
		/*
		 * Manhattan Distance
		 * 
		 * Calling Manhattan method to calculate distance and writing output to file.
		 * 
		 */

		@SuppressWarnings("unchecked")
		void manhattanTrainDistance() throws FileNotFoundException {
			KNN_Distance euclidean = new KNN_Distance();

			Iterator<double[]> trainOverfitITR = trainfeatures.iterator();

			PrintWriter pw = new PrintWriter("ManhattanTrainResult.txt");
			while (trainOverfitITR.hasNext()) {
				double trainOverfitF[] = trainOverfitITR.next();
				Iterator<double[]> trainITR = trainfeatures.iterator();
				int noOfobject = 0;
				ArrayList<DistanceAndFeatures> ts = new ArrayList<>();
				while (trainITR.hasNext()) {
					double trainF[] = trainITR.next();
					double dist = 0;
					dist = euclidean.getManhattanDistance(trainF, trainOverfitF);

					String trainFeat = trainlabel.get(noOfobject);
					DistanceAndFeatures DfObject = new DistanceAndFeatures(dist, trainFeat);
					ts.add(DfObject);
					Collections.sort(ts);
					noOfobject++;

				}

				/*
				 * counting top predicted label based on k value
				 */

				int flag = 0, positive = 0, negative = 0;

				while (flag < knn_value) {
					DistanceAndFeatures s = ts.get(flag);
					String s1 = s.getLabel();
					if (s1.equals("1"))
						positive++;
					else 
						negative++;
					flag++;

				}

				/*
				 * counting label and selecting highest label count as prediction label and
				 * writing to output file.
				 */
				if (positive > negative) {
					pw.println("1");

				} else 
					pw.println("0");
			}
			pw.close();
		}

		
		@SuppressWarnings("unchecked")
		void manhattanTestDistance() throws FileNotFoundException {
			KNN_Distance euclidean = new KNN_Distance();

			Iterator<double[]> testITR = testfeatures.iterator();

			PrintWriter pw = new PrintWriter("ManhattanTestResult.txt");

			while (testITR.hasNext()) {
				double testF[] = testITR.next();
				Iterator<double[]> trainITR = trainfeatures.iterator();
				int noOfobject = 0;
				ArrayList<DistanceAndFeatures> ts = new ArrayList<>();
				while (trainITR.hasNext()) {
					double trainF[] = trainITR.next();
					double dist = 0;
					dist = euclidean.getManhattanDistance(trainF, testF);

					String trainFeat = trainlabel.get(noOfobject);
					DistanceAndFeatures DfObject = new DistanceAndFeatures(dist, trainFeat);
					ts.add(DfObject);
					Collections.sort(ts);
					noOfobject++;

				}

				/*
				 * counting top predicted label based on k value
				 */

				int flag = 0, positive = 0, negative = 0;

				while (flag < knn_value) {
					DistanceAndFeatures s = ts.get(flag);
					String s1 = s.getLabel();
					if (s1.equals("1"))
						positive++;
					else 
						negative++;
					flag++;

				}

				/*
				 * counting label and selecting highest label count as prediction label and
				 * writing to output file.
				 */
				if (positive > negative) {
					pw.println("1");

				} else 
					pw.println("0");
			}
			pw.close();
		}

		
		/*
		 * method to get K value and Distance metrics.
		 */

		void getKValueandDistMetrics() {

			System.out.println("Enter the K value of KNN ");
			knn_value = sc.nextInt();
			// Restricted k value less 30.
			if (knn_value > 30) {
				System.out.println("K Value is out of range.");
				getKValueandDistMetrics();
			} else {

				System.out.println(
						"Select below distance metric(1 or 2)\n1 Euclidean Distance Metrics\n2 Manhattan Distance Metrics");
				DistanceMetricsSelection = sc.nextInt();

			}

		}

		/*
		 * Calculating accuracy for model based Euclidean and Manhattan distance.
		 */
		void accuracy() throws IOException {
			int count = 0;
			File fileTest = null;
			File fileTrain = null;
			if (DistanceMetricsSelection == 1) {
				fileTest = new File("EuclideanTestResult.txt");
				fileTrain = new File("EuclideanTrainResult.txt");
			}

			else if (DistanceMetricsSelection == 2) {
				fileTest = new File("ManhattanTestResult.txt");
				fileTrain = new File("ManhattanTrainResult.txt");

			}

			BufferedReader rf = new BufferedReader(new FileReader(fileTest));
			BufferedReader label = new BufferedReader(new FileReader(new File("RealTestLabel.txt")));
			String s = rf.readLine();
			while (s != null) {
				String lab = label.readLine();
				if (s.equals(lab)) {

				} else {
					count++;
				}

				s = rf.readLine();
			}

			System.out.println("Test Accuracy is: " + ((float) 100 - (((float) count / totalNumberOfLabel) * 100)) + "%");
			rf.close();
			label.close();
//			
			BufferedReader rf2 = new BufferedReader(new FileReader(fileTrain));
			BufferedReader label2 = new BufferedReader(new FileReader(new File("RealTrainLabel.txt")));
			String s2 = rf2.readLine();
			while (s2 != null) {
				String lab = label2.readLine();
				if (s2.equals(lab)) {

				} else {
					count++;
				}

				s2 = rf2.readLine();
			}

			System.out.println("Train Accuracy is: " + ((float) 100 - (((float) count / totalNumberOfLabel) * 100)) + "%");
			rf2.close();
			label2.close();
		}

}
