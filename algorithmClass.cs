using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace algorithmClass
{
    public class testAlgorithm
    /*  Class for handing test algorithms. */
    {
        public Func<string, double> method;
        public IDictionary<string, double> results = null;
        public string name;

        public testAlgorithm(Func<string, double> method, string name)
        /*  Method refers to the algorithm that must be executed through
            this class, and name is a arbitrary string used for identification
            purposes. */
        {
            this.method = method;
            this.name = name;
        }

        public void executeAlgorithm(string input)
        /*  Executes the stored function with the given input, and stores
            the result in the internal dictionary. */
        {
            // Keep dictionary up-to-date with all the results from the current algorithm
            if (this.results == null)
            {
                this.results = new Dictionary<string, double>();
                this.results.Add(new KeyValuePair<string, double>(input, this.method(input)));
            }
            else
            {
                this.results.Add(new KeyValuePair<string, double>(input, this.method(input)));
            }
        }
    }
}
