import java.io.*;
import java.util.*;

public class DiabetesDetect {
	
	private double rate;

	/** the weight to learn */
	private double[] weights;

	/** the number of iterations */
	private int ITERATIONS = 3000;

	public DiabetesDetect(int n) {
		this.rate = 0.0001;
		weights = new double[n];
		
	}

	private static double sigmoid(double z) {
		double retVal = (1.0 / (1.0 + Math.exp(z)));
		return retVal;
	}

	public void train(List<Instance> instances) {
		for (int n=0; n<ITERATIONS; n++) {
			for (int i=0; i<instances.size(); i++) {
				double[] x = instances.get(i).x;
				double predicted = classify(x);
				int label = instances.get(i).label;
				for (int j=0; j<weights.length; j++) {
					weights[j] = weights[j] - ((rate * (predicted - label) * x[j])/768);
				}

			}
	    	System.out.println("iteration: " + n + " " + Arrays.toString(weights));
		}
	}
	

	private double classify(double[] x) {
		double logit = 0.0;
		for (int i=0; i<weights.length;i++)  {
			logit += weights[i] * x[i];
		}
		return sigmoid(logit);
	}
	public static class Instance {
		public int label;
		public double[] x;

		public Instance(int label, double[] x) {
			this.label = label;
			this.x = x;
		}
	}
	
	public static List<Instance> readDataSet(String file)throws FileNotFoundException {
		ArrayList<Instance> dataset = new ArrayList<Instance>();
		Scanner inputStream = null;
		try {
			inputStream = new Scanner(new File(file));
			inputStream.next();
			int j = 0;
			String data = " ";
			data = inputStream.next();
			while(inputStream.hasNext()) {
				String[] val = data.split(",");
				double[] temp = new double[8];
				//temp[0] = 1;
				int label = 0;
				if(j == 8) {
					label = Integer.parseInt(val[j]);
					j = 0;
					data = inputStream.next();
				}
				
				temp[j] = Double.parseDouble(val[j]);
				j++;
				Instance instance = new Instance(label, temp);
				dataset.add(instance);
		}
		}
		finally {
			if (inputStream != null)
				inputStream.close();
		}
		return dataset;
	}


	public static void main(String... args) throws FileNotFoundException {
		List<Instance> instances = readDataSet("/home/namanb99/Desktop/MINOR_2/diabetes.csv");
		//System.out.println(instances);
		DiabetesDetect logistic = new DiabetesDetect(8);
		logistic.train(instances);
		double[] x = {1,103,30,38,83,43.3,0.183,33};
		if(logistic.classify(x)>0.5) {
		System.out.println("prob(1|x) = 1");
		}
		else {
			System.out.println("prob(1|x) = 0");
		}
}
}
