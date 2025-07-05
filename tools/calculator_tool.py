"""
简单计算器工具：表达式求值
"""
import re
import math
import ast
import operator
from typing import Dict, Any, Union


class CalculatorTool:
    """计算器工具，支持安全的数学表达式计算"""
    
    def __init__(self):
        """初始化计算器"""
        self.enabled = True
        
        # 支持的运算符
        self.operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        
        # 支持的函数
        self.functions = {
            'abs': abs,
            'round': round,
            'max': max,
            'min': min,
            'sum': sum,
            'sqrt': math.sqrt,
            'pow': pow,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'pi': math.pi,
            'e': math.e,
        }
    
    def calculate(self, expression: str) -> Dict[str, Any]:
        """
        计算数学表达式
        
        Args:
            expression: 数学表达式字符串
            
        Returns:
            计算结果字典，包含result和error信息
        """
        try:
            print(f"🧮 计算表达式: {expression}")
            
            # 清理表达式
            clean_expr = self._clean_expression(expression)
            
            if not clean_expr:
                return {
                    'result': None,
                    'error': '表达式为空或无效',
                    'expression': expression
                }
            
            # 尝试使用AST安全计算
            try:
                result = self._safe_eval(clean_expr)
                print(f"✅ 计算结果: {result}")
                return {
                    'result': result,
                    'error': None,
                    'expression': clean_expr
                }
            except Exception as e:
                # 如果AST失败，尝试简单的eval（有限的安全性）
                result = self._simple_eval(clean_expr)
                return {
                    'result': result,
                    'error': None,
                    'expression': clean_expr
                }
                
        except Exception as e:
            print(f"❌ 计算出错: {str(e)}")
            return {
                'result': None,
                'error': f'计算错误: {str(e)}',
                'expression': expression
            }
    
    def _clean_expression(self, expression: str) -> str:
        """
        清理和标准化表达式
        
        Args:
            expression: 原始表达式
            
        Returns:
            清理后的表达式
        """
        if not expression:
            return ""
        
        # 移除多余的空格
        expr = re.sub(r'\s+', '', expression)
        
        # 替换常见的中文数学符号
        replacements = {
            '×': '*',
            '÷': '/',
            '（': '(',
            '）': ')',
            '，': ',',
        }
        
        for old, new in replacements.items():
            expr = expr.replace(old, new)
        
        # 移除不安全的字符
        allowed_chars = set('0123456789+-*/().%^,abcdefghijklmnopqrstuvwxyz')
        expr = ''.join(c for c in expr if c.lower() in allowed_chars)
        
        # 替换^为**（Python的幂运算）
        expr = expr.replace('^', '**')
        
        return expr
    
    def _safe_eval(self, expression: str) -> float:
        """
        使用AST安全计算表达式
        
        Args:
            expression: 要计算的表达式
            
        Returns:
            计算结果
        """
        try:
            # 解析表达式为AST
            node = ast.parse(expression, mode='eval')
            return self._eval_node(node.body)
        except:
            raise ValueError("表达式语法错误")
    
    def _eval_node(self, node) -> Union[int, float]:
        """
        递归计算AST节点
        
        Args:
            node: AST节点
            
        Returns:
            节点计算结果
        """
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.operators.get(type(node.op))
            if op:
                return op(left, right)
            else:
                raise ValueError(f"不支持的运算符: {type(node.op)}")
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.operators.get(type(node.op))
            if op:
                return op(operand)
            else:
                raise ValueError(f"不支持的一元运算符: {type(node.op)}")
        elif isinstance(node, ast.Call):
            func_name = node.func.id if hasattr(node.func, 'id') else None
            if func_name in self.functions:
                args = [self._eval_node(arg) for arg in node.args]
                return self.functions[func_name](*args)
            else:
                raise ValueError(f"不支持的函数: {func_name}")
        elif isinstance(node, ast.Name):
            if node.id in self.functions:
                return self.functions[node.id]
            else:
                raise ValueError(f"未知变量: {node.id}")
        else:
            raise ValueError(f"不支持的节点类型: {type(node)}")
    
    def _simple_eval(self, expression: str) -> float:
        """
        简单的eval计算（受限的安全性）
        
        Args:
            expression: 表达式
            
        Returns:
            计算结果
        """
        # 检查表达式是否只包含安全字符
        safe_chars = set('0123456789+-*/().%')
        if not all(c in safe_chars or c.isspace() for c in expression):
            raise ValueError("表达式包含不安全字符")
        
        # 限制表达式长度
        if len(expression) > 100:
            raise ValueError("表达式过长")
        
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return float(result)
        except:
            raise ValueError("计算失败")
    
    def solve_equation(self, equation: str) -> Dict[str, Any]:
        """
        求解简单方程（x的一元方程）
        
        Args:
            equation: 方程字符串，如 "2*x + 3 = 7"
            
        Returns:
            求解结果
        """
        try:
            if '=' not in equation:
                return {
                    'result': None,
                    'error': '不是有效的方程（缺少等号）',
                    'equation': equation
                }
            
            left, right = equation.split('=', 1)
            left = left.strip()
            right = right.strip()
            
            # 简单的线性方程求解（这里实现一个基础版本）
            # 实际项目中可能需要使用sympy等库
            
            # 尝试几个常见的x值，看哪个使方程成立
            for x in range(-100, 101):
                try:
                    left_val = self._evaluate_with_x(left, x)
                    right_val = self._evaluate_with_x(right, x)
                    
                    if abs(left_val - right_val) < 0.0001:
                        return {
                            'result': x,
                            'error': None,
                            'equation': equation,
                            'verification': f'{left_val} ≈ {right_val}'
                        }
                except:
                    continue
            
            return {
                'result': None,
                'error': '无法求解此方程',
                'equation': equation
            }
            
        except Exception as e:
            return {
                'result': None,
                'error': f'方程求解错误: {str(e)}',
                'equation': equation
            }
    
    def _evaluate_with_x(self, expression: str, x_value: float) -> float:
        """
        将x替换为具体值后计算表达式
        
        Args:
            expression: 包含x的表达式
            x_value: x的值
            
        Returns:
            计算结果
        """
        # 替换x为具体数值
        expr = expression.replace('x', str(x_value))
        result = self.calculate(expr)
        
        if result['error']:
            raise ValueError(result['error'])
        
        return result['result']
    
    def get_tool_info(self) -> Dict[str, Any]:
        """获取工具信息"""
        return {
            'name': 'calculator',
            'description': '执行数学计算和简单方程求解',
            'parameters': {
                'expression': '数学表达式或方程',
                'operation_type': '操作类型：calculate/solve（可选）'
            },
            'example_usage': 'calculator("2 + 3 * 4")',
            'capabilities': [
                '基础数学运算（+、-、*、/、%、**）',
                '数学函数（sin、cos、sqrt、log等）',
                '简单方程求解',
                '安全表达式计算'
            ],
            'supported_functions': list(self.functions.keys())
        }
    
    def format_result(self, result: Dict[str, Any]) -> str:
        """
        格式化计算结果为可读文本
        
        Args:
            result: 计算结果字典
            
        Returns:
            格式化的结果文本
        """
        if result['error']:
            return f"计算错误: {result['error']}"
        
        if result['result'] is None:
            return "无法计算结果"
        
        # 格式化数字显示
        value = result['result']
        if isinstance(value, float):
            if value.is_integer():
                formatted_value = str(int(value))
            else:
                formatted_value = f"{value:.6f}".rstrip('0').rstrip('.')
        else:
            formatted_value = str(value)
        
        return f"{result['expression']} = {formatted_value}"
