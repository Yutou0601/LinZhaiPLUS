using System;
using IronPython.Hosting;
using Microsoft.Scripting;
using Microsoft.Scripting.Hosting;
using System.IO;

namespace CodeF
{
    class Program
    {
        static void Main(string[] args)
        {
            var engine = Python.CreateEngine();

            // 使用相對路徑取得專案根目錄
            string relativePath = @"DatabaseNetwork\Test_GUI_1\TestIPyProject";
            string fullPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, relativePath);

            // 設置 Python 模組路徑
            var searchPaths = engine.GetSearchPaths();
            searchPaths.Add(fullPath);
            engine.SetSearchPaths(searchPaths);

            var scope = engine.CreateScope();
            scope.SetVariable("x", 10);

            // 使用相對路徑指定 Python 文件
            string scriptFile = Path.Combine(fullPath, "pyModule.py");
            var source = engine.CreateScriptSourceFromFile(scriptFile);

            var compilation = source.Compile();
            var result = compilation.Execute(scope); // Execute Python File 

            /*
            foreach(var item in scope.GetVariableNames( )) {
                //Console.WriteLine($"Valiable = {item},Value : {scope.GetVariable(item) } ");
                if (item == "xy") 
                {
                    Console.WriteLine($" xy()  : {scope.GetVariable(item)} ");
                }
            }
            */

            Console.ReadLine();
        }
    }
}
