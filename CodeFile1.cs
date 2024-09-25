using System;
using IronPython.Hosting;
using Microsoft.Scripting;
using Microsoft.Scripting.Hosting;

namespace CodeF
{
    class Program
    {
        static void Main(string[] args)
        {
            var engine = Python.CreateEngine();

            var searchPaths = engine.GetSearchPaths();
            searchPaths.Add(@"C:\Users\user\Desktop\DatabaseNetwork\Test_GUI_1\TestIPyProject\");
            engine.SetSearchPaths(searchPaths);

            var scope = engine.CreateScope();
            scope.SetVariable("x", 10);

            
            var source = engine.CreateScriptSourceFromFile(@"C:\Users\user\Desktop\DatabaseNetwork\Test_GUI_1\TestIPyProject\pyModule.py");

            var compilation = source.Compile();
            var result = compilation.Execute(scope);//Execute Python File 


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