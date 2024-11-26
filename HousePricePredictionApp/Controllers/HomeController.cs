using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using System.IO;

namespace HousePricePredictionApp.Controllers
{
    public class HomeController : Controller
    {
        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public IActionResult PredictPrice(string feature1, string feature2)
        {
            // 构建传递给 Python 脚本的参数
            string args = $"{feature1} {feature2}";

            // 设置 Python 脚本的路径
            string pythonScript = Path.Combine("..", "LINZHAIPLUS", "Library", "Module", "house_price_prediction.py");

            // 创建进程信息
            ProcessStartInfo psi = new ProcessStartInfo();
            psi.FileName = "python";
            psi.Arguments = $"\"{pythonScript}\" {args}";
            psi.UseShellExecute = false;
            psi.RedirectStandardOutput = true;
            psi.RedirectStandardError = true;
            psi.CreateNoWindow = true;

            // 启动进程并读取输出
            using (Process process = Process.Start(psi))
            {
                string output = process.StandardOutput.ReadToEnd();
                string errors = process.StandardError.ReadToEnd();
                process.WaitForExit();

                if (!string.IsNullOrEmpty(errors))
                {
                    // 处理错误
                    ViewBag.Error = errors;
                    return View("Error");
                }

                // 将输出传递给视图
                return View("PredictPrice", output);
            }
        }
    }
}
