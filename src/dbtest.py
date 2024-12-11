import pymysql
# 데이터베이스 연결 정보 설정
db_config = {

}


def fetch_sign_words():
    try:
        # 데이터베이스 연결
        connection = pymysql.connect(**db_config)
        cursor = connection.cursor()

        # SQL 쿼리 실행
        query = "SELECT id, sentence FROM sign_quizzes"
        cursor.execute(query)

        # 결과 가져오기
        results = cursor.fetchall()
        for row in results:
            print(f"ID: {row[0]}, Word: {row[1]}")

    except pymysql.MySQLError as e:
        print(f"Error while connecting to MySQL: {e}")
    finally:
        # 연결 닫기
        if 'connection' in locals() and connection.open:
            cursor.close()
            connection.close()


# 함수 실행
fetch_sign_words()
