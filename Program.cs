using System;
using OpenCvSharp;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Drawing.Printing;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Data.SqlClient;
using Point = OpenCvSharp.Point;
using static System.Net.Mime.MediaTypeNames;
using System.Collections.Specialized;
using System.Diagnostics;
using System.Data.SqlTypes;
using AForge;
using AForge.Imaging;
using AForge.Imaging.Filters;
using System.Runtime.Remoting.Channels;
using static Functions.Functions;

namespace Sharpness
{
    internal class Program
    {
        public class testSet
        {
            public string set;
            public string crop;
            public string uncrop;
            public string background;
            public string name;
            public string results;

            public testSet(string directory, string name)
            {
                this.set = directory;
                this.crop = directory + Path.DirectorySeparatorChar + "cropped";
                this.uncrop = directory + Path.DirectorySeparatorChar + "uncropped";
                this.background = directory + Path.DirectorySeparatorChar + "background.jpg";
                this.name = name;
                this.results = directory + Path.DirectorySeparatorChar + "results";
            }
        }

        static void Main(string[] args)
        {
            // Define all image directories
            string directory = "C:\\Users\\danie\\Desktop\\Sharpness Tests\\Sharpness";
            testSet small_set = new testSet(directory + Path.DirectorySeparatorChar + "small_set", "small");
            testSet set_50_30_045 = new testSet(directory + Path.DirectorySeparatorChar + "50_30_0,45_set", "50");
            testSet set_60_30_045 = new testSet(directory + Path.DirectorySeparatorChar + "60_30_0,45_set", "60");
            testSet set_70_30_045 = new testSet(directory + Path.DirectorySeparatorChar + "70_30_0,45_set", "70");
            testSet[] allSets = new testSet[4] { small_set, set_50_30_045, set_60_30_045, set_70_30_045 };

            bool full_test = false;
            if (full_test)
            {
                // Loop through all testsets
                foreach(testSet set in allSets)
                {
                    Console.WriteLine(set.name);
                    // Run all functions on selected set
                    var results = FunctionTesting.FunctionTesting.testFunction(set);

                    // Print test results
                    foreach (KeyValuePair<string, IDictionary<string, float>> kvp in results)
                    {
                        double total = 0;
                        int i = 0;
                        foreach (KeyValuePair<string, float> kvp2 in kvp.Value)
                        {
                            total += kvp2.Value;
                            i += 1;
                        }
                        Console.WriteLine($"{kvp.Key}: {total / i}");
                    }
                    Console.WriteLine("____________________________________");
                    // Visualise results by sorting them according to given scores
                    FunctionTesting.FunctionTesting.rankResults(results, $"{set.name}_results", directory, set.set);
                }
            }
            else
            {
                // Smaller testset
                var results = FunctionTesting.FunctionTesting.testFunction(small_set);

                // Print test results
                foreach (KeyValuePair<string, IDictionary<string, float>> kvp in results)
                {
                    double total = 0;
                    int i = 0;
                    foreach (KeyValuePair<string, float> kvp2 in kvp.Value)
                    {
                        total += kvp2.Value;
                        i += 1;
                    }
                    Console.WriteLine($"{kvp.Key}: {total / i}");
                }
                Console.WriteLine("____________________________________");
                FunctionTesting.FunctionTesting.rankResults(results, $"{small_set.name}_results", directory, small_set.set);
            }
        }
    }
}


namespace FunctionTesting
{
    public class testAlgorithm
    {
        Func<string, double> method;
        string input;
        double result;
        string name;

        public testAlgorithm(Func<string, double> method, string name)
        {
            this.method = method;
            this.name = name;
        }

        public void executeAlgorithm()
        {
            this.result = this.method(this.input);
        }
    }

    class FunctionTesting
    {
        public static IDictionary<string, IDictionary<string, float>> testFunction(Sharpness.Program.testSet set)
        {
            // Create dictionary to hold all dictionaries
            IDictionary<string, IDictionary<string, float>> results = new Dictionary<string, IDictionary<string, float>>();

            // Create dictionary for each of the tested functions
            IDictionary<string, float> results_1 = new Dictionary<string, float>();
            IDictionary<string, float> results_2 = new Dictionary<string, float>();
            IDictionary<string, float> results_3 = new Dictionary<string, float>();
            IDictionary<string, float> results_4 = new Dictionary<string, float>();
            IDictionary<string, float> results_5 = new Dictionary<string, float>();
            IDictionary<string, float> results_6 = new Dictionary<string, float>();
            IDictionary<string, float> results_7 = new Dictionary<string, float>();
            IDictionary<string, float> results_8 = new Dictionary<string, float>();
            IDictionary<string, float> results_9 = new Dictionary<string, float>();
            IDictionary<string, float> results_10 = new Dictionary<string, float>();
            IDictionary<string, float> results_11 = new Dictionary<string, float>();
            IDictionary<string, float> results_12 = new Dictionary<string, float>();
            IDictionary<string, float> results_13 = new Dictionary<string, float>();

            // Run functions on each of the images found in the given directory
            foreach (string uncropped in Directory.GetFiles(set.uncrop))
            {
                string imageDst = set.crop + Path.DirectorySeparatorChar + uncropped.Split(Path.DirectorySeparatorChar).Last();
                imageDst = imageDst.Replace("Uncropped", "Cropped");
                cropImage(uncropped, imageDst, set.background);
                Console.WriteLine(1);
                results_1.Add(imageDst, Functions.Functions.laplace(imageDst));
                Console.WriteLine(2);
                results_2.Add(imageDst, Functions.Functions.blurLaplace(imageDst));
                Console.WriteLine(3);
                results_3.Add(imageDst, Functions.Functions.blurLaplaceSum(imageDst));
                Console.WriteLine(4);
                results_4.Add(imageDst, Functions.Functions.blurLaplaceVar(imageDst));
                Console.WriteLine(5);
                results_5.Add(imageDst, Functions.Functions.specialSobel(imageDst));
                Console.WriteLine(6);
                results_6.Add(imageDst, Functions.Functions.baseSobel(imageDst));
                Console.WriteLine(7);
                results_7.Add(imageDst, Functions.Functions.tenegradSobel(imageDst));
                Console.WriteLine(8);
                results_8.Add(imageDst, Functions.Functions.sobelVariance(imageDst));
                Console.WriteLine(9);
                results_9.Add(imageDst, Functions.Functions.canny(imageDst));
                Console.WriteLine(10);
                results_10.Add(imageDst, Functions.Functions.greyVariance(imageDst));
                Console.WriteLine(11);
                results_11.Add(imageDst, Functions.Functions.edgeWidth(imageDst));
            }
            // These functions crop images using a different method
            foreach (string uncropped in Directory.GetFiles(set.uncrop))
            {
                string imageDst = set.crop + Path.DirectorySeparatorChar + uncropped.Split(Path.DirectorySeparatorChar).Last();
                Console.WriteLine(12);
                results_12.Add(imageDst, Functions.Functions.existingSharpness(uncropped, set.background));
                Console.WriteLine(13);
                results_13.Add(imageDst, Functions.Functions.existingOldSharpness(uncropped, set.background));
            }
            // Store all found results in main dictionary
            results.Add("laplace", results_1);
            results.Add("blurLaplace", results_2);
            results.Add("blurLaplaceSum", results_3);
            results.Add("blurLaplaceVar", results_4);
            results.Add("specialSobel", results_5);
            results.Add("baseSobel", results_6);
            results.Add("tenegradSobel", results_7);
            results.Add("sobelVariance", results_8);
            results.Add("canny", results_9);
            results.Add("greyVariance", results_10);
            results.Add("edgeWidth", results_11);
            results.Add("existingSharpness", results_12);
            results.Add("existingOldSharpness", results_13);
            return results;
        }
    
        public static void rankResults(IDictionary<string, IDictionary<string, float>> results, string name, string directory, string images)
        {
            // Function for creating new folders with images for each algorithm
            // These images are the same ones that have been tested, but now rearranged/renamed to
            //  reflect their ranking by the algorihm
            foreach (KeyValuePair<string, IDictionary<string, float>> kvp in results)
            {
                // Create new folder for selected algorithm
                string newDir = directory + $"\\{name}\\{kvp.Key}";
                if (Directory.Exists(newDir))
                {
                    Directory.Delete(newDir, true);
                }
                Directory.CreateDirectory(newDir);
                var sortedDict = kvp.Value.OrderBy(x => x.Value).ToDictionary(x => x.Key, x => x.Value);
                int i = 0;
                foreach (KeyValuePair<string, float> kvp2 in sortedDict)
                {
                    string imageLoc = newDir + $"\\{i}-{kvp2.Value}.jpg";
                    string oldImageLoc = kvp2.Key.Replace("Uncropped", "Cropped");
                    oldImageLoc = oldImageLoc.Replace("uncropped", "cropped");
                    File.Copy(oldImageLoc, imageLoc);
                    i++;
                }
            }
        }
    
        public static void cropImage(string image, string dst, string background, int threshold=4)
        {
            // Source is old branch-code - CytoCV.cs
            // Grayscale the images
            Mat backgroundImage = Cv2.ImRead(background);
            Cv2.CvtColor(backgroundImage, backgroundImage, ColorConversionCodes.RGB2GRAY);
            Mat src = Cv2.ImRead(image);
            Cv2.CvtColor(src, src, ColorConversionCodes.RGB2GRAY);

            // Create buffer for storing temporary results
            Mat buffer = new Mat(backgroundImage.Size(), backgroundImage.Type());
            // Create kernel for morphological dilation
            Mat kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(5, 5));

            // Calculate and store absolute difference between fore- and background
            Cv2.Absdiff(src, backgroundImage, buffer);
            // Finds global minimum and maximum absolute differences
            Cv2.MinMaxIdx(buffer, out double min, out double max);

            // Morphologically erode, and then dilate, image; removes noise
            Cv2.MorphologyEx(buffer, buffer, MorphTypes.Open, kernel);
            // Apply threshold on Absdiff-image
            // Threshold equals to the minimum Absdiff + 1/4th of the difference between the
            //  minimum and maximum Absdiffs
            Cv2.Threshold(buffer, buffer, min + (max - min) / threshold, 255, ThresholdTypes.Binary);

            // Finds contours in binary buffer-image
            var contours = Cv2.FindContoursAsArray(buffer, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            // Calculate the minimal boundingbox in which the image-particle fits
            // Loop through all detected contours, and enlarge box to fit all of them
            Rect boundingBox = new Rect();
            foreach (var contour in contours)
            {
                if (boundingBox.Width == 0)
                    boundingBox = Cv2.BoundingRect(contour);
                else
                    boundingBox |= Cv2.BoundingRect(contour);
            }
            // If there is no boundingbox, there are no contours
            if (boundingBox.Width == 0)
            {
                // Option 1: retry with lower threshold to have an easier time finding particles
                //  Sometimes results in larger images that take a helluva lot longer to process
                cropImage(image, dst, background, threshold + 1);
                // Option 2: do not crop image
                //  Same issue as option 1, but worse
                //src.SaveImage(dst);
                // Option 3: ignore image, and do not process it using the test-algorithms
                //  No long processing times, but also less images on which to base observations
                return;
            }
            // Else
            // Increase boundingbox by 40px on each side
            boundingBox.Inflate(40, 40);

            // Create image boundary the size of the background image
            Rect imageBounds = new Rect(new Point(0, 0), src.Size());
            // Crop boundingbox to imageBounds limits
            // I assume this is incase the +40 inflation causes the image to become
            //  larger than the original
            boundingBox &= imageBounds;

            // Crop image according to boundingbox
            // Copped image contains cropped version of original image
            var croppedImage = new Mat(src, boundingBox);
            croppedImage.SaveImage(dst);

            return;
        }
    }
}


namespace Functions
{
    internal class Functions
    {
        // Laplace focused
        public static float laplace(string image)
        {
            // https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
            // https://stackoverflow.com/questions/58005091/how-to-get-the-variance-of-laplacian-in-c-sharp
            // Grayscale
            Mat image_color = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(image_color, src, ColorConversionCodes.BGR2GRAY);

            Mat dst = new Mat();
            // Apply Lacplacian for edge detection
            Cv2.Laplacian(src, dst, MatType.CV_64F);
            // Get standard deviation of laplacian
            Cv2.MeanStdDev(dst, out var mean, out var stddev);
            return (float)(stddev.Val0 * stddev.Val0);
        }

        public static float blurLaplace(string image)
        {
            // Grayscale
            Mat image_color = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(image_color, src, ColorConversionCodes.BGR2GRAY);

            // Apply median blur for image smoothing/removing noise
            var medianBlurredImage = src.MedianBlur(5);
            // Apply laplacian for edge detection
            var laplacianImage = medianBlurredImage.Laplacian(MatType.CV_8U, 5);
            // Get standard deviation of laplacian
            Cv2.MeanStdDev(laplacianImage, out Scalar _, out Scalar sd);
            return (float)(sd[0] * sd[0]);
        }

        public static float blurLaplaceSum(string image)
        {
            // Source: https://sci-hub.wf/10.1109/icpr.2000.903548
            // Grayscale
            Mat image_color = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(image_color, src, ColorConversionCodes.BGR2GRAY);

            // Apply median blur for image smoothing/removing noise
            var medianBlurredImage = src.MedianBlur(5);
            // Apply laplacian for edge detection
            var laplacianImage = medianBlurredImage.Laplacian(MatType.CV_8U, 5);

            return (float)Cv2.Sum(laplacianImage);
        }

        public static float blurLaplaceVar(string image)
        {
            // Source: https://sci-hub.wf/10.1109/icpr.2000.903548
            // Grayscale
            Mat image_color = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(image_color, src, ColorConversionCodes.BGR2GRAY);

            // Apply median blur for image smoothing/removing noise
            var medianBlurredImage = src.MedianBlur(5);
            // Apply laplacian for edge detection
            var laplacianImage = medianBlurredImage.Laplacian(MatType.CV_8U, 5);
            Bitmap lpImage = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(laplacianImage);
            // Get mean of laplacian
            double mean = 0;
            for (int x = 0; x < lpImage.Width; x++)
            {
                for (int y = 0; y < lpImage.Height; y++)
                {
                    mean += lpImage.GetPixel(x, y).R;
                }
            }
            mean = mean / (lpImage.Width * lpImage.Height);

            // Calculate variance
            double globalVar = 0;
            for(int x = 0; x < lpImage.Width; x++)
            {
                for(int y = 0; y < lpImage.Height; y++)
                {
                    globalVar += Math.Pow(lpImage.GetPixel(x, y).R - mean, 2);
                }
            }
            return (float)globalVar;
        }

        // Sobel focused
        public static float specialSobel(string image)
        {
            // Grayscale
            Mat image_color = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(image_color, src, ColorConversionCodes.BGR2GRAY);

            // Apply median blur
            var medianBlurredImage = src.MedianBlur(5);

            // Convert to bitmap for custom Sobel operation
            Bitmap bitImage = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(medianBlurredImage);

            // Create sobel kernels
            double[,] gx = new double[,] { { -Math.Sqrt(2) / 4, 0, Math.Sqrt(2) / 4 }, { -1, 0, 1 }, { -Math.Sqrt(2) / 4, 0, Math.Sqrt(2) / 4 } };
            double[,] gy = new double[,] { { Math.Sqrt(2) / 4, 1, -Math.Sqrt(2) / 4 }, { 0, 0, 0 }, { -Math.Sqrt(2) / 4, -1, -Math.Sqrt(2) / 4 } };

            // Apply manual Sobel for edge detection, and calculate sharpness
            double sharpness = 0;
            for(int x = 1; x < bitImage.Width - 1; x++)
            {
                for(int y = 1; y < bitImage.Height - 1; y++)
                {
                    double sumX = 0;
                    double sumY = 0;
                    for(int i = -1; i <= 1; i++)
                    {
                        for(int j = -1; j <= 1; j++)
                        {
                            Color c = bitImage.GetPixel(x + i, y + j);
                            sumX += c.R * gx[i + 1, j + 1];
                            sumY += c.R * gy[i + 1, j + 1];
                        }
                    }
                    sharpness += Math.Sqrt(sumX * sumX + sumY * sumY);
                }
            }
            // Normalise sharpness
            sharpness = sharpness / (bitImage.Width * bitImage.Height);
            
            return (float)sharpness;
        }

        public static float baseSobel(string image)
        {
            // https://stackoverflow.com/questions/48751468/c-sharp-identify-blur-image-with-fft
            // Grayscale
            Mat image_color = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(image_color, src, ColorConversionCodes.BGR2GRAY);

            Mat Gx = new Mat();
            Mat Gy = new Mat();
            // Apply Sobel
            Cv2.Sobel(src, Gx, MatType.CV_32F, 1, 0);
            Cv2.Sobel(src, Gy, MatType.CV_32F, 0, 1);
            // Normalise results
            double normGx = Cv2.Norm(Gx);
            double normGy = Cv2.Norm(Gy);
            double sumSq = normGx * normGx + normGy * normGy;

            return (float)(1.0 / (sumSq / (src.Size().Height * src.Size().Width) + 1e-6));
        }

        public static float tenegradSobel(string image)
        {
            // Source: https://sci-hub.wf/10.1109/icpr.2000.903548
            // Grayscale
            Mat image_color = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(image_color, src, ColorConversionCodes.BGR2GRAY);

            Mat Gx = new Mat();
            Mat Gy = new Mat();
            // Apply Sobel
            Cv2.Sobel(src, Gx, MatType.CV_8U, 1, 0);
            Cv2.Sobel(src, Gy, MatType.CV_8U, 0, 1);
            // Convert to bitmap; normally conversion wouldnt be necessary, however
            //  for some reason, when using .get on Gx/Gy, I get AccessViolationException
            Bitmap GxB = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(Gx);
            Bitmap GyB = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(Gy);

            // Calculate mean and create new matrix that combines the 2 sobels
            double[,] sobel = new double[Gx.Width, Gx.Height];
            double mean = 0;
            for (int x = 0; x < Gx.Width; x++)
            {
                for (int y = 0; y < Gx.Height; y++)
                {
                    sobel[x, y] = Math.Sqrt(Math.Pow(GxB.GetPixel(x, y).R, 2) + Math.Pow(GyB.GetPixel(x, y).R, 2));
                    mean += sobel[x, y];
                }
            }
            mean = mean / (Gx.Width * Gx.Height);

            // Calculate standard deviation
            double stdDev = 0;
            for (int x = 0; x < Gx.Width; x++)
            {
                for (int y = 0; y < Gx.Height; y++)
                {
                    stdDev += Math.Pow(sobel[x, y] - mean, 2);
                }
            }
            stdDev = Math.Sqrt(stdDev / (Gx.Width * Gx.Height));

            // Thresholds
            double Th = mean + 1.6 * stdDev;
            double Tl = Th / 2;

            // Calculate Tenegrad focus measure
            double totalTenegrad = 0;
            for(int x = 0; x < Gx.Width; x++)
            {
                for(int y = 0; y < Gy.Height; y++)
                {
                    if (sobel[x, y] > Tl)
                    {
                        totalTenegrad += sobel[x, y];
                    }
                }
            }

            return (float)totalTenegrad;
        }

        public static float sobelVariance(string image)
        {
            // Source: https://sci-hub.wf/10.1109/icpr.2000.903548
            // Grayscale
            Mat image_color = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(image_color, src, ColorConversionCodes.BGR2GRAY);

            Mat Gx = new Mat();
            Mat Gy = new Mat();
            // Apply Sobel
            Cv2.Sobel(src, Gx, MatType.CV_8U, 1, 0);
            Cv2.Sobel(src, Gy, MatType.CV_8U, 0, 1);
            // Convert to bitmap; normally conversion wouldnt be necessary, however
            //  for some reason, when using .get on Gx/Gy, I get AccessViolationException
            Bitmap GxB = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(Gx);
            Bitmap GyB = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(Gy);

            // Calculate mean and create new matrix that combines the 2 sobels
            double[,] sobel = new double[Gx.Width, Gx.Height];
            double mean = 0;
            for (int x = 0; x < Gx.Width; x++)
            {
                for (int y = 0; y < Gx.Height; y++)
                {
                    sobel[x, y] = Math.Sqrt(Math.Pow(GxB.GetPixel(x, y).R, 2) + Math.Pow(GyB.GetPixel(x, y).R, 2));
                    mean += sobel[x, y];
                }
            }
            mean = mean / (Gx.Width * Gx.Height);

            // Calculate standard deviation
            double stdDev = 0;
            for (int x = 0; x < Gx.Width; x++)
            {
                for (int y = 0; y < Gx.Height; y++)
                {
                    stdDev += Math.Pow(sobel[x, y] - mean, 2);
                }
            }
            stdDev = Math.Sqrt(stdDev / (Gx.Width * Gx.Height));

            // Thresholds
            double Th = mean + 1.6 * stdDev;
            double Tl = Th / 2;

            // Calculate Tenegrad focus measure
            double totalVar = 0;
            for (int x = 0; x < Gx.Width; x++)
            {
                for (int y = 0; y < Gy.Height; y++)
                {
                    if (sobel[x, y] > Tl)
                    {
                        totalVar += Math.Pow(sobel[x, y] - mean, 2);
                    }
                }
            }

            return (float)totalVar;
        }

        // Canny focused
        public static float canny(string image)
        {
            // Grayscale
            Mat image_color = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(image_color, src, ColorConversionCodes.BGR2GRAY);

            // Mean gradient magnitude
            Mat Gx = new Mat();
            Mat Gy = new Mat();
            // Apply Sobel
            Cv2.Sobel(src, Gx, MatType.CV_32F, 1, 0);
            Cv2.Sobel(src, Gy, MatType.CV_32F, 0, 1);
            Mat sobel = Gx + Gy;
            Cv2.ConvertScaleAbs(sobel, sobel);
            // Calculate mean and standard deviation
            Cv2.MeanStdDev(sobel, out Scalar mean, out Scalar stdDev);

            // Thresholds
            double Th = mean[0] + 1.6 * stdDev[0];
            double Tl = Th / 2;

            // Canny
            Mat dst = new Mat();
            Cv2.Canny(src, dst, Tl, Th);

            return (float)Cv2.Sum(dst)[0];
        }

        // Special
        public static float greyVariance(string image)
        {
            // Source: https://sci-hub.wf/10.1109/icpr.2000.903548
            // Grayscale
            Mat image_color = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(image_color, src, ColorConversionCodes.BGR2GRAY);

            // Convert to bitmap for custom Sobel operation
            Bitmap bitImage = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(src);
            float[,] localVars = new float[bitImage.Width, bitImage.Height];

            // Get global mean variance based off of local variances
            float globalMeanVar = 0;
            for(int x = 1; x < bitImage.Width - 1; x++)
            {
                for(int y = 1; y < bitImage.Height - 1; y++)
                {
                    // Calculate local mean
                    float localMean = 0;
                    for(int m = -1; m <= 1; m++)
                    {
                        for(int n = -1; n <= 1; n++)
                        {
                            localMean += bitImage.GetPixel(x + m, y + n).R;
                        }
                    }
                    localMean = localMean / 9;

                    // Calculate local variance
                    float localVar = 0;
                    for (int m = -1; m <= 1; m++)
                    {
                        for (int n = -1; n <= 1; n++)
                        {
                            // Calculate what the new value of the pixel will be
                            // This is the variance between the localMean and existing value of the selected pixel
                            int diff = bitImage.GetPixel(x + m, y + n).R - (int)localMean;
                            localVar += (float)Math.Pow(diff, 2);
                        }
                    }
                    // Store the local variance for later use
                    localVars[x, y] = localVar / 9;
                    globalMeanVar += localVar / 9;
                }
            }
            // Calculate the global mean variance
            globalMeanVar = globalMeanVar / (bitImage.Width - 2 * bitImage.Height - 2);

            // Get variance between global mean variance and local variances
            float globalVar = 0;
            for (int x = 1; x < bitImage.Width - 1; x++)
            {
                for (int y = 1; y < bitImage.Height - 1; y++)
                {
                    globalVar += localVars[x, y] - globalMeanVar;
                }
            }

            return globalVar;
        }

        public static float edgeWidth(string image)
        {
            // Grayscale
            Mat image_color = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(image_color, src, ColorConversionCodes.BGR2GRAY);

            // Mean gradient magnitude
            Mat Gx = new Mat();
            Mat Gy = new Mat();
            // Apply Sobel
            Cv2.Sobel(src, Gx, MatType.CV_32F, 1, 0);
            Cv2.Sobel(src, Gy, MatType.CV_32F, 0, 1);
            Mat sobel = Gx + Gy;
            sobel.ConvertTo(sobel, MatType.CV_8UC1);

            var lines = Cv2.HoughLines(sobel, 1, Math.PI / 180, 50);
            int edgeCount = 0;
            int edgeWidth = 0;

            foreach(var line in lines)
            {
                Point p1 = new Point();
                Point p2 = new Point();
                double a = Math.Cos(line.Theta);
                double b = Math.Sin(line.Theta);
                double x0 = a * line.Rho;
                double y0 = b * line.Rho;
                p1.X = (int)(x0 + 1000 * (-b));
                p1.Y = (int)(y0 + 1000 * (a));
                p2.X = (int)(x0 - 1000 * (-b));
                p2.Y = (int)(y0 - 1000 * (a));

                // Calculate width of edge segment
                int width = (int)Math.Sqrt(Math.Pow(p1.X - p2.X, 2) + Math.Pow(p1.Y - p2.Y, 2));
                edgeCount++;
                edgeWidth += width;
            }

            return (float)edgeWidth / edgeCount;
        }

        // Existing pieces of code
        public class existing_CytoCropResult
        {
            public int ImageID;
            public Mat Image;
            public float Sharpness;
            public Rectangle BoundingBox;

            public existing_CytoCropResult()
            {
                ImageID = -1;
                Image = null;
                Sharpness = 0;
            }
        }

        public static float existingSharpness(string image, string background)
        {
            // Source is old branch-code - CytoCV.cs
            existing_CytoCropResult src = existingCrop(image, background);
            // Apply Median filter using kernel size of 5
            // Kernel size of 5 probably means a 5x5 kernel
            if (src.Image == null)
            {
                return (float)0.0;
            }
            var medianBlurredImage = src.Image.MedianBlur(5);
            // Apply Laplacian using kernel size of 5
            // The CV_8U, for argument ddepth, defines the type of data that is stored,
            //  in this Unsigned Char; It has nothing to do with the shape of the matrix
            var laplacianImage = medianBlurredImage.Laplacian(MatType.CV_8U, 5);
            // Computes mean value (_) and standard deviation (sd) for image
            Cv2.MeanStdDev(laplacianImage, out _, out Scalar sd);

            // Calculates and stores the sharpness of the image
            // croppedImage > medianBlur > Laplace > Standard deviation squared
            src.Sharpness = (float)(sd[0] * sd[0]);

            return src.Sharpness;
        }

        public static float existingOldSharpness(string image, string background)
        {
            // Source is old local sharpness test code
            existing_CytoCropResult src = existingCrop(image, background);
            // Apply Median filter using kernel size of 5
            // Kernel size of 5 probably means a 5x5 kernel
            if(src.Image == null)
            {
                return (float)0.0;
            }
            var medianBlurredImage = src.Image.MedianBlur(5);
            // Apply Laplacian using kernel size of 5
            // The CV_8U, for argument ddepth, defines the type of data that is stored,
            //  in this Unsigned Char; It has nothing to do with the shape of the matrix
            var laplacianImage = medianBlurredImage.Laplacian(MatType.CV_32F, 3);
            // Computes mean value (_) and standard deviation (sd) for image
            Cv2.MeanStdDev(laplacianImage, out _, out Scalar sd);

            // Calculates and stores the sharpness of the image
            // croppedImage > medianBlur > Laplace > Standard deviation squared
            src.Sharpness = (float)(sd[0] * sd[0]);

            return src.Sharpness;
        }

        public static existing_CytoCropResult existingCrop(string image, string background)
        {
            // Source is old branch-code - CytoCV.cs
            Mat backgroundImage = Cv2.ImRead(background);
            Cv2.CvtColor(backgroundImage, backgroundImage, ColorConversionCodes.RGB2GRAY);
            Mat src = Cv2.ImRead(image);
            Cv2.CvtColor(src, src, ColorConversionCodes.RGB2GRAY);
            // Create image boundary the size of the background image
            Rect imageBounds = new Rect(new Point(0, 0), backgroundImage.Size());
            // Create buffer for storing temporary results
            Mat buffer = new Mat(backgroundImage.Size(), backgroundImage.Type());
            // Create kernel for morphological dilation
            Mat kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(5, 5));

            // Calculate and store absolute difference between fore- and background
            Cv2.Absdiff(src, backgroundImage, buffer);
            // Finds global minimum and maximum absolute differences
            Cv2.MinMaxIdx(buffer, out double min, out double max);

            // Apply threshold on Absdiff-image
            // Threshold equals to the minimum Absdiff + 1/4th of the difference between the
            //  minimum and maximum Absdiffs
            Cv2.Threshold(buffer, buffer, min + (max - min) / 4, 255, ThresholdTypes.Binary);
            // Morphologically erode, and then dilate, image; removes noise
            Cv2.MorphologyEx(buffer, buffer, MorphTypes.Open, kernel);

            // Finds contours in binary buffer-image
            var contours = Cv2.FindContoursAsArray(buffer, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            // Calculate the minimal boundingbox in which the image-particle fits
            // Loop through all detected contours, and enlarge box to fit all of them
            Rect boundingBox = new Rect();
            foreach (var contour in contours)
            {
                if (boundingBox.Width == 0)
                    boundingBox = Cv2.BoundingRect(contour);
                else
                    boundingBox |= Cv2.BoundingRect(contour);
            }
            // If there is no boundingbox, there are no contours, thus there's no particle
            if (boundingBox.Width == 0)
            {
                return new existing_CytoCropResult();
            }
            // Else
            var cropResult = new existing_CytoCropResult();

            // Increase boundingbox by 40px on each side
            boundingBox.Inflate(40, 40);
            // Crop boundingbox to imageBounds limits
            // I assume this is incase the +40 inflation causes the image to become
            //  larger than the original
            boundingBox &= imageBounds;

            // Crop image according to boundingbox
            // Copped image contains cropped version of original image
            var croppedImage = new Mat(src, boundingBox);
            // Stores the new BoundingBox of the image
            cropResult.BoundingBox = new Rectangle(boundingBox.X, boundingBox.Y, boundingBox.Width, boundingBox.Height);
            // Stores the cropped image itself
            cropResult.Image = croppedImage;

            return cropResult;
        }
    }
}
