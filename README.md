using System;
using System.IO;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using OpenCvSharp;
using OpenCvSharp.Dnn;

using System.Windows;
namespace TestOpencv
 {
 public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }
               
        int m_Count = 0;
 private void button_findJPG_Click(object sender, EventArgs e)
     {
        string path = @"D:\AI MODEL\model.pb";
            var net = CvDnn.ReadNetFromTensorflow(path);

            DirectoryInfo dir_raw = new DirectoryInfo(@"D:\TEMP\越南\result");
            DirectoryInfo dir_output = new DirectoryInfo(@"D:\TEMP\越南");

            System.Collections.IEnumerator myEnum = dir_raw.GetFiles("*.jpg").GetEnumerator();


            int iHeight = 128;
            int iWidth = 128;
            double dRescale = 1;

            while (myEnum.MoveNext())
            {
                String fiName = dir_raw + "\\" + myEnum.Current.ToString();

                //加了後面的格式設定就會有問題
                Mat source = Cv2.ImRead(fiName);

                if (source.Empty())
                    continue;

                Mat convert_img = new Mat();


                if (source.Cols == iWidth && source.Rows == iHeight)
                {
                    source.ConvertTo(convert_img, MatType.CV_32FC3, 1 / dRescale);
                }
                else
                {
                    Mat resize_Img = new Mat();
                    Cv2.Resize(source, resize_Img, new OpenCvSharp.Size(iHeight, iWidth));
                    resize_Img.ConvertTo(convert_img, MatType.CV_32FC3, 1 / dRescale);
                }
                                              
                Mat blob = CvDnn.BlobFromImage(convert_img);

                net.SetInput(blob);

                Mat result = net.Forward();              

                OpenCvSharp.Point MinPoint, MaxPoint;
                Cv2.MinMaxLoc(result.Reshape(1, 1), out MinPoint, out MaxPoint);

                int m = MaxPoint.X;

                String name = myEnum.Current.ToString();

                String foName = dir_output + "\\" + m.ToString() + "\\";

                if (Directory.Exists(foName))
                {
                    System.IO.File.Copy(fiName, foName + name);
                }
                label_Count.Text = name;
                
                Application.DoEvents();

            }

            GC.Collect();
            MessageBox.Show("OK");
          }
           private void button_convert_to_gray_Click(object sender, EventArgs e)
        {
            DirectoryInfo dir_raw = new DirectoryInfo(@"D:\TEMP\越南\result");
            DirectoryInfo dir_output = new DirectoryInfo(@"D:\TEMP\越南\result");

            System.Collections.IEnumerator myEnum = dir_raw.GetFiles("*.jpg").GetEnumerator();
            
           int m_Count = 0;
            
           while (myEnum.MoveNext())
            {
                String fiName = dir_raw + "\\" + myEnum.Current.ToString();               
              
                String name = myEnum.Current.ToString();

                String foName = dir_output + "\\" ;

                if (Directory.Exists(foName))
                {

                 //加了後面的格式設定就會有問題
                 Mat source = Cv2.ImRead(fiName);

                 if (source.Empty())
                  
                        continue;

                 Mat convert_gray = new Mat();

                 Cv2.CvtColor(source, convert_gray, ColorConversionCodes.BGR2GRAY);

                 foName = dir_output + "\\" +m_Count.ToString() + ".jpg";

                 m_Count++;

                 Cv2.ImWrite(foName, convert_gray);

                 }

                label_Count.Text = name;
                Application.DoEvents();

           }

         GC.Collect();
         MessageBox.Show("OK");
        }
      }
     }
