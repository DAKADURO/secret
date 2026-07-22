from src.motor_rag import MotorRAG
motor = MotorRAG()
for chunk in motor.generar_diagnostico_stream('que servicio y que elementos se remplazan a las 4000 horas', 'KAISHAN', 'KRSD-125'):
  print(chunk, end='')
