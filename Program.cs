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
using testSet = setClass.testSet;
using testAlgorithm = algorithmClass.testAlgorithm;

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
            string directory = "C:\\Users\\danie\\Desktop\\Sharpness Tests\\Sharpness";
            testSet small_set = new testSet(directory + Path.DirectorySeparatorChar 
                + "small_set", "small");
            testSet set_50_30_045 = new testSet(directory + Path.DirectorySeparatorChar 
                + "50_30_0,45_set", "50");
            testSet set_60_30_045 = new testSet(directory + Path.DirectorySeparatorChar 
                + "60_30_0,45_set", "60");
            testSet set_70_30_045 = new testSet(directory + Path.DirectorySeparatorChar 
                + "70_30_0,45_set", "70");
            testSet[] allSets = new testSet[4] { small_set, set_50_30_045, set_60_30_045, 
                set_70_30_045 };

            // Select whether to run a quick test, or not
            bool full_test = false;
            if (full_test)
            {
                // Loop through all testsets
                foreach(testSet set in allSets)
                {
                    Console.WriteLine(set.name);
                    // Run all functions on selected set
                    FunctionTesting.FunctionTesting.testFunction(set);

                    // Loop through all tested algorithms
                    foreach (var algo in small_set.algorithms)
                    {
                        // Calculate average score of the algorithm for this set
                        double total = 0;
                        int i = 0;
                        foreach (KeyValuePair<string, double> kvp in algo.results)
                        {
                            total += kvp.Value;
                            i += 1;
                        }
                        Console.WriteLine($"{algo.name}: {total / i}");
                    }
                    Console.WriteLine("____________________________________");
                    // Visualise results by sorting them according to given scores
                    FunctionTesting.FunctionTesting.rankResults(set);
                }
            }
            else
            {
                // Smaller testset
                FunctionTesting.FunctionTesting.testFunction(small_set);

                // Loop through all tested algorithms
                foreach (var algo in small_set.algorithms)
                {
                    // Calculate average score of the algorithm for this set
                    double total = 0;
                    int i = 0;
                    foreach (KeyValuePair<string, double> kvp in algo.results)
                    {
                        total += kvp.Value;
                        i += 1;
                    }
                    Console.WriteLine($"{algo.name}: {total / i}");
                }
                Console.WriteLine("____________________________________");
                // Visualise results by sorting them according to given scores
                FunctionTesting.FunctionTesting.rankResults(small_set);
            }
        }
    }
}


namespace FunctionTesting
{
    class FunctionTesting
    {
        public static void testFunction(testSet set)
        /*  This function tests all created functions using the given testset */
        {
            // Create list for all the to be executed algorithms
            testAlgorithm al1 = new testAlgorithm(laplace, "laplace");
            testAlgorithm al2 = new testAlgorithm(blurLaplace, "blurLaplace");
            testAlgorithm al3 = new testAlgorithm(blurLaplaceSum, "blurLaplaceSum");
            testAlgorithm al4 = new testAlgorithm(blurLaplaceVar, "blurLaplaceVar");
            testAlgorithm al5 = new testAlgorithm(specialSobel, "specialSobel");
            testAlgorithm al6 = new testAlgorithm(baseSobel, "baseSobel");
            testAlgorithm al7 = new testAlgorithm(tenegradSobel, "tenegradSobel");
            testAlgorithm al8 = new testAlgorithm(sobelVariance, "sobelVariance");
            testAlgorithm al9 = new testAlgorithm(canny, "canny");
            testAlgorithm al10 = new testAlgorithm(greyVariance, "greyVariance");
            testAlgorithm al11 = new testAlgorithm(edgeWidth, "edgeWidth");
            testAlgorithm al12 = new testAlgorithm(existingSharpness, "existingSharpness");
            testAlgorithm al13 = new testAlgorithm(existingOldSharpness, "existingOldSharpness");

            testAlgorithm[] allAlgo = new testAlgorithm[13] { al1, al2, 
                al3, al4, al5, al6, al7, al8, al9, al10, al11, al12, al13 };

            // Loop through all images in given set
            foreach (string uncropped in Directory.GetFiles(set.uncrop))
            {
                // Crop selected image
                string imageDst = set.crop + Path.DirectorySeparatorChar + 
                    uncropped.Split(Path.DirectorySeparatorChar).Last();
                imageDst = imageDst.Replace("Uncropped", "Cropped");
                cropImage(uncropped, imageDst, set.background);
                // Loop through all selected functions
                foreach(var algo in allAlgo)
                {
                    Console.WriteLine(algo.name);
                    algo.executeAlgorithm(imageDst);
                }
            }
            // Store algorithms and their results within given set
            set.algorithms = allAlgo;
        }
    
        public static void rankResults(testSet set)
        /*  Function for creating new folders with images for each tested
            algorithm. Theses images are the same ones that have been tested,
            but have now been rearranged and renamed to reflect their ranking
            by the algorithm. */
        {
            // Loop through all tested algorithms
            foreach (var algo in set.algorithms)
            {
                // Create new folder for selected algorithm
                string newDir = set.results + Path.DirectorySeparatorChar + 
                    $"{algo.name}";
                if (Directory.Exists(newDir))
                {
                    Directory.Delete(newDir, true);
                }
                Directory.CreateDirectory(newDir);
                // Sort algorithm results by value
                var sortedDict = algo.results.OrderBy(x => x.Value).
                    ToDictionary(x => x.Key, x => x.Value);

                // Loop through the sorted images within the dictionary
                int i = 0;
                foreach (KeyValuePair<string, double> kvp in sortedDict)
                {
                    // Rename and store the selected image
                    string imageLoc = newDir + Path.DirectorySeparatorChar + 
                        $"{i}-{kvp.Value}.jpg";
                    File.Copy(kvp.Key, imageLoc);
                    i++;
                }
            }
        }
    
        public static void cropImage(string image, string dst, string background, 
            int threshold=4)
        /*  This function crops given images to a (hopefully) smaller size.
            By vaguely detecting where within the image the particle is located,
            a smaller, quicker to analyse, version of the original image is 
            cropped out around the particle. */
        {
            // Based on old branch-code in CytoCV.cs
            // Grayscale the images
            Mat backgroundImage = Cv2.ImRead(background);
            Cv2.CvtColor(backgroundImage, backgroundImage, 
                ColorConversionCodes.RGB2GRAY);
            Mat src = Cv2.ImRead(image);
            Cv2.CvtColor(src, src, ColorConversionCodes.RGB2GRAY);

            // Create buffer for storing temporary results
            Mat buffer = new Mat(backgroundImage.Size(), backgroundImage.Type());
            // Create kernel for morphological dilation
            Mat kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, 
                new OpenCvSharp.Size(5, 5));

            // Calculate and store absolute difference between fore- and background
            Cv2.Absdiff(src, backgroundImage, buffer);
            // Find global minimum and maximum absolute differences
            Cv2.MinMaxIdx(buffer, out double min, out double max);

            // Morphologically erode, and then dilate, image; removes noise
            Cv2.MorphologyEx(buffer, buffer, MorphTypes.Open, kernel);
            // Apply threshold on Absdiff-image
            // Threshold is normally 4, but can vary if 4 is too high to detect
            //  a particle with
            Cv2.Threshold(buffer, buffer, min + (max - min) / threshold, 255, 
                ThresholdTypes.Binary);

            // Finds contours in thresholded image
            var contours = Cv2.FindContoursAsArray(buffer, RetrievalModes.External, 
                ContourApproximationModes.ApproxSimple);

            // Calculate the minimal boundingbox in which the image-particle fits
            Rect boundingBox = new Rect();
            // Loop through all detected contours, and update box to fit all of them
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
                // Option 1: retry with lower threshold to have an easier time
                //  finding particles; Sometimes results in larger images that 
                //  take a helluva lot longer to process
                cropImage(image, dst, background, threshold + 1);
                // Option 2: do not crop image
                //  Same issue as option 1, but worse
                //src.SaveImage(dst);
                // Option 3: ignore image, and do not process it using the
                //  test-algorithms; No long processing times, but also less
                //  images on which to base observations
                return;
            }
            // Increase boundingbox by 40px on each side
            // This is done incase there are tiny bits of the particle that
            //  may otherwise fall outside of the boundingBox
            boundingBox.Inflate(40, 40);

            // Create image boundary the size of the background image
            Rect imageBounds = new Rect(new Point(0, 0), src.Size());
            // Crop boundingbox to imageBounds limits
            // I assume this is incase the +40 inflation causes the image to 
            //  become larger than the original; under any normal circumstances,
            //  this should never be the case. If something weird happens,
            //  however, this failsafe might be useful in avoiding out-of-bounds
            //  errors.
            boundingBox &= imageBounds;

            // Crop image according to boundingbox
            // Copped image contains cropped version of original image
            var croppedImage = new Mat(src, boundingBox);
            croppedImage.SaveImage(dst);
        }
    }
}


namespace Functions
{
    internal class Functions
    {
        // Laplace focused
        public static double laplace(string image)
        /*  Applies the standard laplacian function, without
            blurring, editing, or anything else. The return
            value consists of the squared standard deviation
            of this laplacian. */
        {
            // Source: https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
            // Source :https://stackoverflow.com/questions/58005091/how-to-get-the-variance-of-laplacian-in-c-sharp
            // Grayscale
            Mat image_color = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(image_color, src, ColorConversionCodes.BGR2GRAY);

            Mat dst = new Mat();
            // Apply Lacplacian for edge detection
            Cv2.Laplacian(src, dst, MatType.CV_64F);
            // Get standard deviation of laplacian
            Cv2.MeanStdDev(dst, out var _, out var stddev);
            return stddev.Val0 * stddev.Val0;
        }

        public static double blurLaplace(string image)
        /*  Applies laplacian function after first blurring the
            given image using a median filter. The return value
            consists of the squared standard deviation of this
            laplacian. */
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
            return sd[0] * sd[0];
        }

        public static double blurLaplaceSum(string image)
        /*  This function applies the laplacian method after blurring
            the given image using a median filter, and then returns
            the sum of the values within the resulting laplacian image. */
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

            return Cv2.Sum(laplacianImage)[0];
        }

        public static double blurLaplaceVar(string image)
        /*  This function applies the laplacian method after blurring
            the given image using a median filter. Afterwards it calculates
            the mean of the laplacian, and using this mean, the variance
            of the laplacian is returned. */
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
            Bitmap lpImage = OpenCvSharp.Extensions.BitmapConverter.
                ToBitmap(laplacianImage);
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
                    globalVar += Math.Pow(lpImage.GetPixel(x, y).R - 
                        mean, 2);
                }
            }
            // Is the variance not literally Math.Sqrt(stddev)?
            return globalVar;
        }

        // Sobel focused
        public static double specialSobel(string image)
        /*  This function applies the sobel method, using a special
            custom-made kernel/mask. The retuned sharpness-value is
            calculated by averaging the combined Sqrt of the results
            of both sobel operations. */
        {
            // Grayscale
            Mat image_color = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(image_color, src, ColorConversionCodes.BGR2GRAY);

            // Apply median blur
            var medianBlurredImage = src.MedianBlur(5);

            // Convert to bitmap for custom Sobel operation
            Bitmap bitImage = OpenCvSharp.Extensions.BitmapConverter.
                ToBitmap(medianBlurredImage);

            // Create sobel kernels
            double[,] gx = new double[,] { 
                { -Math.Sqrt(2) / 4, 0, Math.Sqrt(2) / 4 }, 
                { -1, 0, 1 }, 
                { -Math.Sqrt(2) / 4, 0, Math.Sqrt(2) / 4 } };
            double[,] gy = new double[,] { 
                { Math.Sqrt(2) / 4, 1, -Math.Sqrt(2) / 4 }, 
                { 0, 0, 0 }, 
                { -Math.Sqrt(2) / 4, -1, -Math.Sqrt(2) / 4 } };

            // Apply manual Sobel for edge detection, and calculate
            //  sharpness
            double sharpness = 0;
            for(int x = 1; x < bitImage.Width - 1; x++)
            {
                for(int y = 1; y < bitImage.Height - 1; y++)
                {
                    double sumX = 0;
                    double sumY = 0;
                    // Apply mask to both gx and gy
                    for(int i = -1; i <= 1; i++)
                    {
                        for(int j = -1; j <= 1; j++)
                        {
                            Color c = bitImage.GetPixel(x + i, y + j);
                            sumX += c.R * gx[i + 1, j + 1];
                            sumY += c.R * gy[i + 1, j + 1];
                        }
                    }
                    // Use the resulting values to calculate sharpness
                    sharpness += Math.Sqrt(sumX * sumX + sumY * sumY);
                }
            }
            // Calculate average sharpness per pixel
            sharpness = sharpness / (bitImage.Width * bitImage.Height);
            
            return sharpness;
        }

        public static double baseSobel(string image)
        /*  Function that applies the basic sobel method for detecting
            edges, and uses this result to calculate image sharpness. */
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
            // Combine the square root of the separate normalisations
            double sumSq = normGx * normGx + normGy * normGy;

            // Calculate average per pixel
            return sumSq / (src.Size().Height * src.Size().Width);
        }

        public static double tenegradSobel(string image)
        /*  Apply the tenegrad sobel method. Unlike the regular sobel method,
            this one uses a threshold to determine which values are high enough
            to count for the result. The threshold is determined using the mean
            and standard deviation of a standard sobel result. */
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
            // Convert to bitmap; normally conversion wouldnt be necessary,
            //  however, for some reason, when using .get on Gx/Gy, I get
            //  AccessViolationException. This step bypasses that error
            Bitmap GxB = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(Gx);
            Bitmap GyB = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(Gy);

            // Calculate mean and combine the 2 sobels into a new matrix
            double[,] sobel = new double[Gx.Width, Gx.Height];
            double mean = 0;
            for (int x = 0; x < Gx.Width; x++)
            {
                for (int y = 0; y < Gx.Height; y++)
                {
                    // Combine sobels
                    sobel[x, y] = Math.Sqrt(Math.Pow(GxB.GetPixel(x, y).R, 2) + 
                        Math.Pow(GyB.GetPixel(x, y).R, 2));
                    // Continue calculating the mean
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

            return totalTenegrad;
        }

        public static double sobelVariance(string image)
        /*  This function uses the sobel variance to determine the sharpness
            of an image. Very similar to the tenegrad function, it instead 
            returns the variance, instead of the sum, of the pixels that 
            pass the calculated threshold. */
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
            // Convert to bitmap; normally conversion wouldnt be necessary,
            //  however, for some reason, when using .get on Gx/Gy, I get
            //  AccessViolationException. This step bypasses that error
            Bitmap GxB = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(Gx);
            Bitmap GyB = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(Gy);

            // Calculate mean and combine the 2 sobels into a new matrix
            double[,] sobel = new double[Gx.Width, Gx.Height];
            double mean = 0;
            for (int x = 0; x < Gx.Width; x++)
            {
                for (int y = 0; y < Gx.Height; y++)
                {
                    // Combine sobels
                    sobel[x, y] = Math.Sqrt(Math.Pow(GxB.GetPixel(x, y).R, 2) + 
                        Math.Pow(GyB.GetPixel(x, y).R, 2));
                    // Continue calculating the mean
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

            // Calculate total variance
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

            return totalVar;
        }

        // Canny focused
        public static double canny(string image)
        /*  This function applies the canny method of calculating edges,
            and uses this in concert with a custom threshold, to determine
            the sharpness of an image. */
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
            // TODO: Correctly combine sobels
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

            return Cv2.Sum(dst)[0];
        }

        // Special
        public static double greyVariance(string image)
        /*  This is a function that calculates the image sharpness through
            comparing the local variance in pixel intensity, to the global 
            variance in pixel intensity. */
        {
            // Source: https://sci-hub.wf/10.1109/icpr.2000.903548
            // Grayscale
            Mat image_color = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(image_color, src, ColorConversionCodes.BGR2GRAY);

            // Convert to bitmap for custom Sobel operation
            Bitmap bitImage = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(src);
            // Create matrix for local variance storage
            double[,] localVars = new double[bitImage.Width, bitImage.Height];

            // Get global mean variance based off of local variances
            double globalMeanVar = 0;
            for(int x = 1; x < bitImage.Width - 1; x++)
            {
                for(int y = 1; y < bitImage.Height - 1; y++)
                {
                    // Calculate local mean
                    double localMean = 0;
                    for(int m = -1; m <= 1; m++)
                    {
                        for(int n = -1; n <= 1; n++)
                        {
                            localMean += bitImage.GetPixel(x + m, y + n).R;
                        }
                    }
                    localMean = localMean / 9;

                    // Calculate local variance
                    double localVar = 0;
                    for (int m = -1; m <= 1; m++)
                    {
                        for (int n = -1; n <= 1; n++)
                        {
                            // Calculate local variance between the localMean and
                            //  existing value of the selected pixel
                            localVar += Math.Pow(bitImage.GetPixel(x + m, y + n).R 
                                - localMean, 2);
                        }
                    }
                    // Store the local variance for later use
                    localVars[x, y] = localVar / 9;
                    globalMeanVar += localVar / 9;
                }
            }
            // Calculate the global mean variance within local variances
            globalMeanVar = globalMeanVar / (bitImage.Width - 2 * bitImage.Height - 2);

            // Get variance between global mean variance and local variances
            double globalVar = 0;
            for (int x = 1; x < bitImage.Width - 1; x++)
            {
                for (int y = 1; y < bitImage.Height - 1; y++)
                {
                    globalVar += localVars[x, y] - globalMeanVar;
                }
            }

            return globalVar;
        }

        public static double edgeWidth(string image)
        /*  This function attempts to determine image sharpness through the
            width of the edges within the image. */
        {
            // Based on file:///C:/Users/danie/Downloads/applsci-12-06712.pdf
            // Grayscale
            Mat image_color = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(image_color, src, ColorConversionCodes.BGR2GRAY);
            Mat Gx = new Mat();
            Mat Gy = new Mat();
            // Apply Sobel
            Cv2.Sobel(src, Gx, MatType.CV_32F, 1, 0);
            Cv2.Sobel(src, Gy, MatType.CV_32F, 0, 1);
            // TODO: Correctly combine the sobels
            Mat sobel = Gx + Gy;
            sobel.ConvertTo(sobel, MatType.CV_8UC1);

            // Calculate lines within the image
            var lines = Cv2.HoughLines(sobel, 1, Math.PI / 180, 50);
            int edgeCount = 0;
            int edgeWidth = 0;

            // Loop through each individual line
            foreach(var line in lines)
            {
                // Determine the location of the selected line
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

            // Return average width
            return edgeWidth / edgeCount;
        }

        // Existing pieces of code

        public static double existingSharpness(string image)
        /*  Calculates the sharpness of an image through an already implemented
            method. It's basically the same as blurLaplace. */
        {
            // Source is old branch-code - CytoCV.cs
            // Grayscale
            Mat image_color = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(image_color, src, ColorConversionCodes.BGR2GRAY);

            // Apply Median filter using kernel size of 5
            // Kernel size of 5 probably means a 5x5 kernel
            var medianBlurredImage = src.MedianBlur(5);

            // Apply Laplacian using kernel size of 5
            // The CV_8U, for argument ddepth, defines the type of data that is stored,
            //  in this Unsigned Char; It has nothing to do with the shape of the matrix
            var laplacianImage = medianBlurredImage.Laplacian(MatType.CV_8U, 5);

            // Computes mean value (_) and standard deviation (sd) for image
            Cv2.MeanStdDev(laplacianImage, out _, out Scalar sd);

            return sd[0] * sd[0];
        }

        public static double existingOldSharpness(string image)
        /*  Calculates the sharpness of an image through an older, already
            implemented method. The only difference lies in the laplacian
            typing. */
        {
            // Source is old local sharpness test code
            // Grayscale
            Mat image_color = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(image_color, src, ColorConversionCodes.BGR2GRAY);

            // Apply Median filter using kernel size of 5
            // Kernel size of 5 probably means a 5x5 kernel
            var medianBlurredImage = src.MedianBlur(5);

            // Apply Laplacian using kernel size of 5
            // The CV_8U, for argument ddepth, defines the type of data that is stored,
            //  in this Unsigned Char; It has nothing to do with the shape of the matrix
            var laplacianImage = medianBlurredImage.Laplacian(MatType.CV_32F, 3);

            // Computes mean value (_) and standard deviation (sd) for image
            Cv2.MeanStdDev(laplacianImage, out _, out Scalar sd);

            return sd[0] * sd[0];
        }
    }
}
