from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import base64
import os
import logging
from zhipuai import ZhipuAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 百度AI配置
BAIDU_API_KEY = os.getenv('BAIDU_API_KEY')
BAIDU_SECRET_KEY = os.getenv('BAIDU_SECRET_KEY')
BAIDU_TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
# 添加百度其他识别接口的URL
BAIDU_DISH_DETECT_URL = "https://aip.baidubce.com/rest/2.0/image-classify/v2/dish"
BAIDU_INGREDIENT_DETECT_URL = "https://aip.baidubce.com/rest/2.0/image-classify/v1/classify/ingredient"
BAIDU_GENERAL_DETECT_URL = "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general"

def identify_with_baidu(image_base64, access_token):
    """使用百度多个识别接口进行识别"""
    logger.info("开始食物识别流程")
    
    params = {
        'image': image_base64,
        'access_token': access_token
    }
    
    # 1. 首先尝试菜品识别
    try:
        response = requests.post(BAIDU_DISH_DETECT_URL, data=params)
        result = response.json()
        logger.info(f"菜品识别结果: {result}")
        
        if 'result' in result and len(result['result']) > 0:
            food_info = result['result'][0]
            if food_info['name'] != '非菜':
                return {
                    'name': food_info['name'],
                    'confidence': food_info['probability'],
                    'is_food': True
                }
    except Exception as e:
        logger.error(f"菜品识别出错: {str(e)}")
    
    # 2. 如果不是菜品，尝试食材识别
    try:
        logger.info("尝试食材识别")
        response = requests.post(BAIDU_INGREDIENT_DETECT_URL, data=params)
        result = response.json()
        logger.info(f"食材识别结果: {result}")
        
        if 'result' in result and len(result['result']) > 0:
            food_info = result['result'][0]
            if food_info['name'] != '非果蔬食材':
                return {
                    'name': food_info['name'],
                    'confidence': food_info['score'],
                    'is_food': True
                }
    except Exception as e:
        logger.error(f"食材识别出错: {str(e)}")
    
    # 3. 最后尝试通用物体识别
    try:
        logger.info("尝试通用物体识别")
        response = requests.post(BAIDU_GENERAL_DETECT_URL, data=params)
        result = response.json()
        logger.info(f"通用识别结果: {result}")
        
        if 'result' in result and len(result['result']) > 0:
            # 过滤出可能是食物的结果
            food_keywords = [
                '食物', '水果', '蔬菜', '零食', '饮料', '甜点', '面包', 
                '糕点', '坚果', '干果', '食材', '主食', '小吃', '糖果',
                '瓜', '果', '菜', '肉', '鱼', '虾', '蛋', '奶'
            ]
            
            first_result = result['result'][0]
            keyword = first_result['keyword']
            root = first_result.get('root', '')
            
            # 检查是否是食物相关
            is_food = any(kw in keyword or kw in root for kw in food_keywords)
            
            return {
                'name': keyword,
                'confidence': first_result['score'],
                'is_food': is_food
            }
            
    except Exception as e:
        logger.error(f"通用识别出错: {str(e)}")
    
    raise ValueError("无法识别物体")

# 智谱AI配置
ZHIPU_API_KEY = os.getenv('ZHIPU_API_KEY')
if not ZHIPU_API_KEY:
    raise ValueError("未找到ZHIPU_API_KEY环境变量")

logger.info(f"ZHIPU_API_KEY: {ZHIPU_API_KEY[:10]}...")
client = ZhipuAI(api_key=ZHIPU_API_KEY)

def get_baidu_access_token():
    """获取百度AI访问令牌"""
    params = {
        'grant_type': 'client_credentials',
        'client_id': BAIDU_API_KEY,
        'client_secret': BAIDU_SECRET_KEY
    }
    response = requests.post(BAIDU_TOKEN_URL, params=params)
    return response.json().get('access_token')

def estimate_weight_from_image(image_base64, food_name):
    """使用智谱AI根据图片估算食物重量"""
    try:
        logger.info(f"开始估算食物重量: {food_name}")
        
        response = client.chat.completions.create(
            model="glm-4v",
            messages=[
                {
                    "role": "system",
                    "content": """你是一个食物重量估算专家。请根据图片中食物的大小和类型来估算重量。
                    请遵循以下规则：
                    1. 仔细观察图片中食物的大小、数量和密度
                    2. 考虑食物的类型和特性
                    3. 只返回数字，不要包含任何单位和文字
                    4. 确保返回的重量在合理范围内"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"这是一张{food_name}的图片，请根据图片估算这份食物的重量（克），只返回数字"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
        )
        
        # 打印完整的响应内容
        logger.info(f"AI响应原始内容: {response.choices[0].message.content}")
        
        weight_text = response.choices[0].message.content.strip()
        logger.info(f"提取的文本内容: {weight_text}")
        
        # 提取数字
        digits = ''.join(filter(str.isdigit, weight_text))
        logger.info(f"提取的数字: {digits}")
        
        weight = int(digits or '0')
        logger.info(f"转换后的重量: {weight}")
        
        # 添加合理性检查
        if weight > 1000 or weight == 0:
            logger.warning(f"重量估算异常: {weight}克，将使用默认值")
            logger.warning(f"异常重量对应的原始响应: {weight_text}")
            
            # 根据食物类型返回合理的默认值
            if any(keyword in food_name for keyword in ['饭', '面', '粥']):
                weight = 300
            elif any(keyword in food_name for keyword in ['肉', '鱼', '鸡', '鸭']):
                weight = 200
            elif any(keyword in food_name for keyword in ['菜', '青菜', '生菜']):
                weight = 150
            else:
                weight = 200
                
            logger.info(f"使用默认重量: {weight}")
        
        return weight
        
    except Exception as e:
        logger.error(f"图片重量估算错误: {str(e)}")
        logger.error(f"错误详情: ", exc_info=True)  # 打印完整的错误堆栈
        return 200  # 发生错误时返回默认值

@app.route('/identify', methods=['POST'])
def identify_food():
    logger.info("收到识别请求")
    
    if 'food_image' not in request.files:
        logger.error("没有文件")
        return jsonify({'error': '没有文件'}), 400
    
    file = request.files['food_image']
    if file.filename == '':
        logger.error("文件名为空")
        return jsonify({'error': '没有选择文件'}), 400
    
    try:
        # 读取图片并转换为base64
        image_content = file.read()
        image_base64 = base64.b64encode(image_content).decode('UTF-8')
        
        # 获取访问令牌
        access_token = get_baidu_access_token()
        
        # 使用多重识别方法
        result = identify_with_baidu(image_base64, access_token)
        
        food_name = result['name']
        confidence = result['confidence']
        is_food = result['is_food']
        
        logger.info(f"识别成功: 名称={food_name}, 置信度={confidence}, 是否食物={is_food}")
        
        response_data = {
            'name': food_name,
            'confidence': confidence,
        }
        
        if is_food:
            # 如果是食物，使用图片估算重量
            weight = estimate_weight_from_image(image_base64, food_name)
            response_data['weight'] = weight
        else:
            # 如果不是食物，添加提示信息
            response_data['message'] = "这个不能吃哦"
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"识别物体时发生错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/calories', methods=['GET'])
def get_calories():
    """使用智谱AI估算卡路里"""
    try:
        food_name = request.args.get('foodName')
        weight = request.args.get('weight')
        
        logger.info(f"收到计算卡路里请求: 食物={food_name}, 重量={weight}克")
        
        if not food_name or not weight:
            return jsonify({'error': '参数不完整', 'calories': 0}), 400
        
                # 如果前端传来了不是食物的请求，直接返回错误
        if 'message' in request.args and request.args['message'] == "这个不能吃哦":
            return jsonify({'error': '这不是食物，无法计算卡路里', 'calories': 0}), 400
        
        response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {
                    "role": "system",
                    "content": """你是一个营养专家，请根据食物重量计算卡路里。
                    1. 只返回数字，不要包含任何单位或说明
                    2. 确保返回的卡路里在合理范围内
                    3. 考虑食物的特性和烹饪方式
                    4. 如果不确定，返回相近食物的平均值"""
                },
                {
                    "role": "user",
                    "content": f"请计算{weight}克{food_name}的卡路里含量，只需要返回数字"
                }
            ]
        )
        
        calories_text = response.choices[0].message.content.strip()
        calories = int(''.join(filter(str.isdigit, calories_text)) or '0')
        
        # 添加合理性检查
        if calories > 2000:
            logger.warning(f"卡路里估算异常: {calories}，将使用默认值")
            calories = min(calories, 800)  # 单份食物不太可能超过800卡路里
            
        return jsonify({'calories': calories})
        
    except Exception as e:
        logger.error(f"计算卡路里时发生错误: {str(e)}")
        return jsonify({
            'error': str(e),
            'calories': 0
        }), 500

if __name__ == '__main__':
    port = os.getenv('PORT', 5000)
    app.run(host='0.0.0.0', port=port)