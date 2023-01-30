using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Point = OpenCvSharp.Point;

namespace Sharpness
{
    internal class Functions
    {
        // Laplace focused
        public static double Laplace(string image)
        /*  Applies the standard laplacian function, without
            blurring, editing, or anything else. The return
            value consists of the squared standard deviation
            of this laplacian. */
        {
            // Source: https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
            // Source :https://stackoverflow.com/questions/58005091/how-to-get-the-variance-of-laplacian-in-c-sharp
            // Grayscale
            Mat imageColor = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(imageColor, src, ColorConversionCodes.BGR2GRAY);

            Mat dst = new Mat();
            // Apply Lacplacian for edge detection
            Cv2.Laplacian(src, dst, MatType.CV_64F);
            // Get standard deviation of laplacian
            Cv2.MeanStdDev(dst, out Scalar _, out Scalar stddev);
            return stddev.Val0 * stddev.Val0;
        }

        public static double BlurLaplace(string image)
        /*  Applies laplacian function after first blurring the
            given image using a median filter. The return value
            consists of the squared standard deviation of this
            laplacian. */
        {
            // Grayscale
            Mat imageColor = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(imageColor, src, ColorConversionCodes.BGR2GRAY);

            // Apply median blur for image smoothing/removing noise
            Mat medianBlurredImage = src.MedianBlur(5);
            // Apply laplacian for edge detection
            Mat laplacianImage = medianBlurredImage.Laplacian(MatType.CV_8U, 5);
            // Get standard deviation of laplacian
            Cv2.MeanStdDev(laplacianImage, out Scalar _, out Scalar sd);
            return sd[0] * sd[0];
        }

        public static double BlurLaplaceSum(string image)
        /*  This function applies the laplacian method after blurring
            the given image using a median filter, and then returns
            the sum of the values within the resulting laplacian image. */
        {
            // Source: https://sci-hub.wf/10.1109/icpr.2000.903548
            // Grayscale
            Mat imageColor = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(imageColor, src, ColorConversionCodes.BGR2GRAY);

            // Apply median blur for image smoothing/removing noise
            Mat medianBlurredImage = src.MedianBlur(5);
            // Apply laplacian for edge detection
            Mat laplacianImage = medianBlurredImage.Laplacian(MatType.CV_8U, 5);

            return Cv2.Sum(laplacianImage)[0];
        }

        public static double BlurLaplaceVar(string image)
        /*  This function applies the laplacian method after blurring
            the given image using a median filter. Afterwards it calculates
            the mean of the laplacian, and using this mean, the variance
            of the laplacian is returned. */
        {
            // Source: https://sci-hub.wf/10.1109/icpr.2000.903548
            // Grayscale
            Mat imageColor = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(imageColor, src, ColorConversionCodes.BGR2GRAY);

            // Apply median blur for image smoothing/removing noise
            Mat medianBlurredImage = src.MedianBlur(5);
            // Apply laplacian for edge detection
            Mat laplacianImage = medianBlurredImage.Laplacian(MatType.CV_8U, 5);
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
            for (int x = 0; x < lpImage.Width; x++)
            {
                for (int y = 0; y < lpImage.Height; y++)
                {
                    globalVar += Math.Pow(lpImage.GetPixel(x, y).R -
                        mean, 2);
                }
            }
            // Is the variance not literally Math.Sqrt(stddev)?
            return globalVar;
        }

        // Sobel focused
        public static double SpecialSobel(string image)
        /*  This function applies the sobel method, using a special
            custom-made kernel/mask. The retuned sharpness-value is
            calculated by averaging the combined Sqrt of the results
            of both sobel operations. */
        {
            // Grayscale
            Mat imageColor = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(imageColor, src, ColorConversionCodes.BGR2GRAY);

            // Apply median blur
            Mat medianBlurredImage = src.MedianBlur(5);

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
            for (int x = 1; x < bitImage.Width - 1; x++)
            {
                for (int y = 1; y < bitImage.Height - 1; y++)
                {
                    double sumX = 0;
                    double sumY = 0;
                    // Apply mask to both gx and gy
                    for (int i = -1; i <= 1; i++)
                    {
                        for (int j = -1; j <= 1; j++)
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

        public static double BaseSobel(string image)
        /*  Function that applies the basic sobel method for detecting
            edges, and uses this result to calculate image sharpness. */
        {
            // https://stackoverflow.com/questions/48751468/c-sharp-identify-blur-image-with-fft
            // Grayscale
            Mat imageColor = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(imageColor, src, ColorConversionCodes.BGR2GRAY);

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

        public static double TenegradSobel(string image)
        /*  Apply the tenegrad sobel method. Unlike the regular sobel method,
            this one uses a threshold to determine which values are high enough
            to count for the result. The threshold is determined using the mean
            and standard deviation of a standard sobel result. */
        {
            // Source: https://sci-hub.wf/10.1109/icpr.2000.903548
            // Grayscale
            Mat imageColor = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(imageColor, src, ColorConversionCodes.BGR2GRAY);

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
            for (int x = 0; x < Gx.Width; x++)
            {
                for (int y = 0; y < Gy.Height; y++)
                {
                    if (sobel[x, y] > Tl)
                    {
                        totalTenegrad += sobel[x, y];
                    }
                }
            }

            return totalTenegrad;
        }

        public static double SobelVariance(string image)
        /*  This function uses the sobel variance to determine the sharpness
            of an image. Very similar to the tenegrad function, it instead 
            returns the variance, instead of the sum, of the pixels that 
            pass the calculated threshold. */
        {
            // Source: https://sci-hub.wf/10.1109/icpr.2000.903548
            // Grayscale
            Mat imageColor = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(imageColor, src, ColorConversionCodes.BGR2GRAY);

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
        public static double Canny(string image)
        /*  This function applies the canny method of calculating edges,
            and uses this in concert with a custom threshold, to determine
            the sharpness of an image. */
        {
            // Grayscale
            Mat imageColor = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(imageColor, src, ColorConversionCodes.BGR2GRAY);

            // Mean gradient magnitude
            Mat Gx = new Mat();
            Mat Gy = new Mat();
            // Apply Sobel
            Cv2.Sobel(src, Gx, MatType.CV_32F, 1, 0);
            Cv2.ConvertScaleAbs(Gx, Gx);
            Cv2.Sobel(src, Gy, MatType.CV_32F, 0, 1);
            Cv2.ConvertScaleAbs(Gy, Gy);
            // Combine sobels
            Mat sobel = Gx + Gy;

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

        public static double CannyMean(string image)
        /*  This function applies the canny method of calculating edges,
            and uses this in concert with a custom threshold, to determine
            the sharpness of an image. Instead of just summing the canny
            results however, this function calculates the mean of these
            results. */
        {
            // Grayscale
            Mat imageColor = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(imageColor, src, ColorConversionCodes.BGR2GRAY);

            // Mean gradient magnitude
            Mat Gx = new Mat();
            Mat Gy = new Mat();
            // Apply Sobel
            Cv2.Sobel(src, Gx, MatType.CV_32F, 1, 0);
            Cv2.ConvertScaleAbs(Gx, Gx);
            Cv2.Sobel(src, Gy, MatType.CV_32F, 0, 1);
            Cv2.ConvertScaleAbs(Gy, Gy);
            // Combine sobels
            Mat sobel = Gx + Gy;

            // Calculate mean and standard deviation
            Cv2.MeanStdDev(sobel, out Scalar mean, out Scalar stdDev);

            // Thresholds
            double Th = mean[0] + 1.6 * stdDev[0];
            double Tl = Th / 2;

            // Canny
            Mat dst = new Mat();
            Cv2.Canny(src, dst, Tl, Th);
            Cv2.MeanStdDev(dst, out Scalar dst_mean, out Scalar _);

            return dst_mean[0] * dst_mean[0];
        }

        // Special
        public static double GreyVariance(string image)
        /*  This is a function that calculates the image sharpness through
            comparing the local variance in pixel intensity, to the global 
            variance in pixel intensity. */
        {
            // Source: https://sci-hub.wf/10.1109/icpr.2000.903548
            // Grayscale
            Mat imageColor = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(imageColor, src, ColorConversionCodes.BGR2GRAY);

            // Convert to bitmap for custom Sobel operation
            Bitmap bitImage = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(src);
            // Create matrix for local variance storage
            double[,] localVars = new double[bitImage.Width, bitImage.Height];

            // Get global mean variance based off of local variances
            double globalMeanVar = 0;
            for (int x = 1; x < bitImage.Width - 1; x++)
            {
                for (int y = 1; y < bitImage.Height - 1; y++)
                {
                    // Calculate local mean
                    double localMean = 0;
                    for (int m = -1; m <= 1; m++)
                    {
                        for (int n = -1; n <= 1; n++)
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

        public static double EdgeWidth(string image)
        /*  This function attempts to determine image sharpness through the
            width of the edges within the image. */
        {
            // Based on file:///C:/Users/danie/Downloads/applsci-12-06712.pdf
            // Grayscale
            Mat imageColor = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(imageColor, src, ColorConversionCodes.BGR2GRAY);
            Mat Gx = new Mat();
            Mat Gy = new Mat();
            // Apply Sobel
            Cv2.Sobel(src, Gx, MatType.CV_32F, 1, 0);
            Cv2.Sobel(src, Gy, MatType.CV_32F, 0, 1);
            // Combine sobels
            Mat sobel = new Mat();
            Cv2.Magnitude(Gx, Gy, sobel);
            Cv2.ConvertScaleAbs(sobel, sobel);

            // Calculate lines within the image
            LineSegmentPolar[] lines = Cv2.HoughLines(sobel, 1, Math.PI / 180, 50);
            int edgeCount = 0;
            int edgeWidth = 0;

            // Loop through each individual line
            foreach (LineSegmentPolar line in lines)
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

        public static double ExistingSharpness(string image)
        /*  Calculates the sharpness of an image through an already implemented
            method. It's basically the same as blurLaplace. */
        {
            // Source is old branch-code - CytoCV.cs
            // Grayscale
            Mat imageColor = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(imageColor, src, ColorConversionCodes.BGR2GRAY);

            // Apply Median filter using kernel size of 5
            // Kernel size of 5 probably means a 5x5 kernel
            Mat medianBlurredImage = src.MedianBlur(5);

            // Apply Laplacian using kernel size of 5
            // The CV_8U, for argument ddepth, defines the type of data that is stored,
            //  in this Unsigned Char; It has nothing to do with the shape of the matrix
            Mat laplacianImage = medianBlurredImage.Laplacian(MatType.CV_8U, 5);

            // Computes mean value (_) and standard deviation (sd) for image
            Cv2.MeanStdDev(laplacianImage, out _, out Scalar sd);

            return sd[0] * sd[0];
        }

        public static double ExistingOldSharpness(string image)
        /*  Calculates the sharpness of an image through an older, already
            implemented method. The only difference lies in the laplacian
            typing. */
        {
            // Source is old local sharpness test code
            // Grayscale
            Mat imageColor = Cv2.ImRead(image);
            Mat src = new Mat();
            Cv2.CvtColor(imageColor, src, ColorConversionCodes.BGR2GRAY);

            // Apply Median filter using kernel size of 5
            // Kernel size of 5 probably means a 5x5 kernel
            Mat medianBlurredImage = src.MedianBlur(5);

            // Apply Laplacian using kernel size of 5
            // The CV_8U, for argument ddepth, defines the type of data that is stored,
            //  in this Unsigned Char; It has nothing to do with the shape of the matrix
            Mat laplacianImage = medianBlurredImage.Laplacian(MatType.CV_32F, 3);

            // Computes mean value (_) and standard deviation (sd) for image
            Cv2.MeanStdDev(laplacianImage, out _, out Scalar sd);

            return sd[0] * sd[0];
        }
    }
}
