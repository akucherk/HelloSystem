using Encog;
using Encog.Engine.Network.Activation;
using Encog.ML.Data.Basic;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using System;

namespace HelloSystem
{
	class Program
	{
		/// <summary>
		/// Input for the XOR function
		/// </summary>
		public static double[][] XORInput =
		{
			new[] { 0.0, 0.0 },
			new[] { 1.0, 0.0 },
			new[] { 0.0, 1.0 },
			new[] { 1.0, 1.0 }
		};

		/// <summary>
		/// Ideal output for the XOR function
		/// </summary>
		public static double[][] XORIdeal =
		{
			new[] { 0.0 },
			new[] { 1.0 },
			new[] { 1.0 },
			new[] { 0.0 }
		};

		static void Main(string[] args)
		{
			var network = new BasicNetwork();
			network.AddLayer(new BasicLayer(null, true, 2));
			network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 3));
			network.AddLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
			network.Structure.FinalizeStructure();
			network.Reset();

			var trainingSet = new BasicMLDataSet(XORInput, XORIdeal);
			var train = new ResilientPropagation(network, trainingSet);
			do
			{
				train.Iteration();
			} while (train.Error > 0.01);

			train.FinishTraining();

			foreach (var pair in trainingSet)
			{
				var output = network.Compute(pair.Input);
				Console.WriteLine(pair.Input[0] + @", " + pair.Input[1] + @" , actual=" + output[0] + @", ideal=" + pair.Ideal[0]);
			}

			EncogFramework.Instance.Shutdown();
			Console.ReadLine();
		}
	}
}