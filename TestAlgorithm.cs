using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgorithmClass
{
    public class TestAlgorithm
    /*  Class for handling test algorithms. */
    {
        private Func<string, double> _method;
        public IDictionary<string, double> Results { get; set; }
        public string Name { get; }
        private Stopwatch _stopWatch { get; set; }
        public double TimeSpan { get; set; }

        public TestAlgorithm(Func<string, double> method, string name)
        /*  Method refers to the algorithm that must be executed through
            this class, and name is a arbitrary string used for identification
            purposes. */
        {
            this._method = method;
            this.Name = name;
            this.Results = null;
        }

        public void ExecuteAlgorithm(string input)
        /*  Executes the stored function with the given input, and stores
            the result in the internal dictionary. */
        {
            // Start stopwatch
            this._stopWatch = new Stopwatch();
            this._stopWatch.Start();
            // Keep dictionary up-to-date with all the results from the current algorithm
            if (this.Results == null)
            {
                this.Results = new Dictionary<string, double> {
                    { input, this._method(input) }
                };
            }
            else
            {
                this.Results.Add(input, this._method(input));
            }
            // Stop stopwatch, and add time to internal counter
            this._stopWatch.Stop();
            this.TimeSpan += this._stopWatch.Elapsed.TotalMilliseconds;
            Console.WriteLine(this._stopWatch.Elapsed.TotalMilliseconds);
        }
    }
}
