package NN;


import com.dkriesel.snipe.core.NeuralNetwork;
import com.dkriesel.snipe.core.NeuralNetworkDescriptor;
import com.dkriesel.snipe.training.TrainingSampleLesson;
import com.dkriesel.snipe.training.ErrorMeasurement;


public class NetWork {

	public NetWork(){
		NeuralNetworkDescriptor desc = new NeuralNetworkDescriptor(16,4,16);
		desc.setSettingsTopologyFeedForward();
		
		NeuralNetwork netz = new NeuralNetwork(desc);
		double[][] in = new double[][]{
				{1, 1, 0, 1},
				{1, 1, 0, 0}
		};
		double[][] out = new double[][]{
				{1, 1, 1, 1},
				{1, 1, 1, 0}
		};
		TrainingSampleLesson lesson = new TrainingSampleLesson(in,out );
		
		System.out.println(ErrorMeasurement.getErrorRootMeanSquareSum(netz, lesson));
		netz.trainBackpropagationOfError(lesson, 100000, 0.02);
		
	}
	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		NeuralNetworkDescriptor desc = new NeuralNetworkDescriptor(2,4,1);
		desc.setSettingsTopologyFeedForward();
		
		NeuralNetwork netz = new NeuralNetwork(desc);
		double[][] in = new double[][]{
				{0, 0},
				{1, 0},
				{0, 1},
				{1, 1},
		};
		double[][] out = new double[][]{
				{0},
				{1},
				{1},
				{0}
		};
		TrainingSampleLesson lesson = new TrainingSampleLesson(in,out );
		
		System.out.println(ErrorMeasurement.getErrorRootMeanSquareSum(netz, lesson));
		netz.trainBackpropagationOfError(lesson, 99999999, 0.02);
		System.out.println(ErrorMeasurement.getErrorRootMeanSquareSum(netz, lesson));
		double[] erg  = netz.propagate(new double[]{1,1});
		
		System.out.println("ergebnisse:");
		
		for(double i : erg){
			System.out.println(i);
		}
	}

}
