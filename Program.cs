using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using Point = OpenCvSharp.Point;

namespace Sharpness
{
    internal class Program
    {
        static void Main(string[] args)
        /*  This is the main function,
            it initialises all of the desired testsets, passes them along to be analysed,
            and provides a display of the results in both numerical, and visual, ways */
        {
            // Define all image directories
            string directory = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.Parent.FullName;
            TestSet smallSet = new TestSet(directory + Path.DirectorySeparatorChar 
                + "small_set", "small");
            TestSet set_50_30_045 = new TestSet(directory + Path.DirectorySeparatorChar 
                + "50_30_0,45_set", "50");
            TestSet set_60_30_045 = new TestSet(directory + Path.DirectorySeparatorChar 
                + "60_30_0,45_set", "60");
            TestSet set_70_30_045 = new TestSet(directory + Path.DirectorySeparatorChar 
                + "70_30_0,45_set", "70");
            TestSet[] allSets = new TestSet[3] { set_50_30_045, set_60_30_045, 
                set_70_30_045 };

            // Select whether to run a quick test, or not
            bool fullTest = false;
            if (fullTest)
            {
                // Loop through all testsets
                foreach(TestSet set in allSets)
                {
                    Console.WriteLine(set.Name);
                    // Run all functions on selected set
                    FunctionTesting.TestFunction(set);

                    // Loop through all tested algorithms
                    foreach (TestAlgorithm algo in set.Algorithms)
                    {
                        // Calculate average score of the algorithm for this set
                        double total = 0;
                        int i = algo.Results.Count;
                        // If there are results to be displayed, do so
                        if (algo.Results.Count >= 1)
                        {
                            foreach (KeyValuePair<string, double> kvp in algo.Results)
                            {
                                total += kvp.Value;
                            }
                        }
                        Console.WriteLine($"{algo.Name}: {algo.TimeSpan / i}; {total / i}");
                    }
                    Console.WriteLine("____________________________________");
                    // Visualise results by sorting them according to given scores
                    FunctionTesting.RankResults(set);
                }
            }
            else
            {
                // Smaller testset
                FunctionTesting.TestFunction(smallSet);

                // Loop through all tested algorithms
                foreach (TestAlgorithm algo in smallSet.Algorithms)
                {
                    // Calculate average score of the algorithm for this set
                    double total = 0;
                    int i = algo.Results.Count;
                    // If there are results to be displayed, do so
                    if (algo.Results.Count >= 1)
                    {
                        foreach (KeyValuePair<string, double> kvp in algo.Results)
                        {
                            total += kvp.Value;
                        }
                    }
                    Console.WriteLine($"{algo.Name}: {algo.TimeSpan / i}; {total / i}");
                }
                Console.WriteLine("____________________________________");
                // Visualise results by sorting them according to given scores
                FunctionTesting.RankResults(smallSet);
            }
        }
    }
}
