# 05장 : 클래스 개요

- **객체 지향 프로그래밍 (OOP : Object Oriented Programming)**에서 가장 많이 사용하는 용어는 의외로 *객체(Object)*가 아닌 *클래스(Class)* 이다.

  

- 일반적으로 객체를 생성할 때 개발 언어에서 제공하는 클래스만을 사용하지 않는다.
  -> 객체 지향 언어는 언어의 문법을 사용하여 클래스를 제작하고, 클래스를 통해 객체를 생성시킬 수 있을 뿐만 아니라 다른 언어로 제작된 라이브러리나 바이너리 형식의 실행 프로그램으로도 객체를 생성시킬 수 있다.

  

- 왜 클래스 지향이 아닌 객체 지향?
  어떠한 방식을 사용했던지간에 객체가 생성되면 우리는 원하는 작업을 수행할 수 있다. 즉 클래스는 객체를 만드는 수단일 뿐, 실제로 우리가 사용하는 것은 객체이므로 객체 지향적 프로그래밍이 더 맞는 말이다.

  

- 왜 클래스를 사용하는가?
  클래스와 객체의 관계는 쉽게 말해 붕어빵 틀과 붕어빵이다.
  클래스가 붕어빵 틀이라고 했을 때, 그 빵틀에 어떤 재료를 어떤 목적으로 넣느냐에 따라 다른 붕어빵을 만들 수 있다(팥을 넣으면 팥 붕어빵, 슈크림을 넣으면 슈크림 붕어빵 ...).

  *과학은 모든 사물을 추상화하고 일반화시키는 작업을 한다* 라는 문장처럼, 우리가 사용할 객체를 추상화하여 클래스로 묶고, 실제로 객체를 사용할 때에는 클래스 안에서 객체를 일반화시켜 생성한다.

  

- 클래스를 어떻게 상용하는가?
  객체 지향 언어를 사용하여 애플리케이션을 개발하는 개발자가 **가장 먼저** 하는 작업은 *사물을 추상화하고, 속성과 기능을 분류하여 클래스를 만드는 일* 이다.
  이 작업은 아주 중요하기때문에 팀으로 나누어 작업을 진행하고, (1)디자인 패턴이나 (2)UML(Unified Modeling Language)를 비롯한 많은 도구를 사용하기도 한다.

  대표적인 UML 다이어그램은 다음과 같다.

  | UML 다이어그램                     | 설명                                                         |
  | ---------------------------------- | ------------------------------------------------------------ |
  | 행위 다이어그램(Behavior Diagram)  | 시스템 내의 객체들의 동적인 행위(Dynamic Behavior)를 보여주며, 시간의 변화에 따른 시스템의 연속된 변경을 설명한다. (ex) Use Case, State Machine, Activity ... |
  | 구조 다이어그램(structure Diagram) | 시스템의 정적 구조(Static Structure)와 시스템의 구성 요소, 서로 다른 구성 요소들 간의 관계를 볼 수 있다. (ex) Class, Object, Composite Structure, Deployment ... |

  > UML Diagram 종류 - https://roynus.tistory.com/1037
  >
  > 구조 다이어그램 참고자료 - https://mongyang.tistory.com/55
  >
  > Use Case Diagram 참고자료 - https://googry.tistory.com/2
  >
  > 디자인 패턴 참고자료 - https://gmlwjd9405.github.io/2018/07/06/design-pattern.html



## 5.1 클래스 포맷

- 프로그램 내부에서 클래스와 구조체를 사용하기 전에 반드시 선언을 해야 한다.

  ```c++
  struct 구조체-이름 final(옵션) [ : 기본-구조체, ...]{
      멤버
  }
  ```

  또는

  ```c++
  class 클래스-이름 final(옵션) [ : 기본-클래스, ...] {
      멤버
  }
  ```

- 클래스-이름(또는 구조체-이름)은 함수 이름처럼 다음과 같은 규칙을 따른다.

  - 첫 번째 문자는 반드시 일반 문자를 사용하거나 또는 밑줄 문자(_)를 사용해야 한다.
  - 첫 번째 문자 이후에 나오는 문자는 일반 문자와 숫자, 그리고 밑줄 문자를 사용할 수 있다.
  - 클래스-이름은 키워드나 지정자와 같은 이름을 사용할 수 없다.
  - 클래스-이름을 명시하지 않는다면 이름이 없는 무명 클래스가 된다.

  > (참고) 클래스 이름은 일반적으로 Person 이나 Car 처럼 명사를 사용한다.
  > final 키워드는 옵션으로 "해당 클래스는 더이상 상속을 허용하지 않는 최종 클래스" 라는 사실을 컴파일러에 알리기 위해 사용한다. 그에 따라 만약 해당 클래스를 상속받는 클래스를 선언한다면 에러가 발생된다.
  > 콜론(':') 기호는 영역을 구분하는 옵션으로, 클래스와 구조체가 상속받는 기본 클래스 또는 기본 구조체를 명시하기 위해 사용한다.



### 클래스의 상속 : 

    1. 클래스의 상속은 앞서 제작한 클래스가 제공하는 멤버 변수와 멤버 함수 등을 포함하여 모든 것을 하위 클래스에 물려줌으로써 프로그램 코드의 재사용을 가능하게 할 뿐만 아니라 차후에 소프트웨어의 유지보수도 간편하게 한다.
    2. C++은 클래스를 상속받을때 한 개 이상의 클래스를 사용할 수 있다(JAVA와 C#과 달리).

  ```c++
  class Rectangle {
  ....
  };
  
  class Clickable {
  ....
  };
  
  // Button 클래스는 Rectangle 클래스와 Clickable 클래스로부터 상속받는다.
  class Button : public Rectangle, public Clickable {
  ....
  };
  ```



- 클래스의 멤버( : 클래스를 구성하는 변수, 상수, 함수, 내부 클래스, 데이터 타입)는 전체적인 가독성을 높이기 위해 선언문과 정의문을 구분한다.

  1. 선언문 : 클래스가 제공하는 멤버의 종류를 언급
  2. 정의문 : 세부적인 멤버 함수의 본문을 구성, 변수의 초기화를 구현

  

- 함수 멤버의 정의문 : 
  함수 멤버의 정의문은 다음과 같이 사용한다.

  ```
  반환-타입 네임스페이스::클래스-이름::함수-이름 (인수) {
  함수-본문
  }
  ```

  > 클래스의 선언문과 정의문이 같은 네임스페이스 내에 존재하면 네임스페이스를 생략할 수 있다.

  

- 클래스 멤버 : 
  클래스는 다음과 같은 멤버를 가질 수 있다.

  1. 함수(Function)
  2. 데이터 타입(사용자가 별도 정의한 데이터 타입)
  3. 생성자(Constructor)
  4. 소멸자(Destructors)
  5. 내부 클래스 또는 내부 구조체
  6. 연산자 오버로딩(Operators Overloading)
  7. 변수(variables)와 상수(Constants)



### 생성자(Constructor)

- 반환되는 데이터 타입없이 클래스 이름과 동일한 이름을 갖고 있는 함수를 생성자라고 한다.

- 타입이 변수 또는 객체로 생성될 때 호출되는 특수 목적의 함수이다.

- 생성자의 포맷 : 

  ```
  지정자 클래스-이름(인수-리스트) noexcept [ : 멤버-초기화-리스트] { 본문 }
  ```

  > 지정자는 옵션으로 explicit 지정자만을 사용할 수 있다. 이는 다음장에서 설명한다.
  > noexcept 지정자 역시 옵션으로 예외의 발생을 허용하지 않는다는 의미로 사용된다.
  > 위의 콜론(':') 기호는 (생성자)와 (초기화시키려는 멤버 변수)나 (별도 호출하고 자 하는 함수, 혹은 다른 생성자)를 구분하는 역할을 한다.
  > 멤버-초기화-리스트(Member Initializer Lists)는 멤버 변수를 초기화시키거나 별도 호출하고 자 하는 함수와 생성자를 호출할 수 있는 기능을 제공한다.
  > 초기화-리스트를 생략한다면 콜론 기호 역시 생략한다.

- 생성자 함수의 목적 : 
  생성자는 다음과 같은 목적으로 사용하는 특별한 종류의 함수이다.

  1. 생성자는 클래스로부터 인스턴스를 생성하고 내부 변수를 초기화시키는 목적으로 사용한다.

     > 인스턴스 : 선언문에 의해 생성된 클래스형의 변수를 인스턴스라 하며, 클래스가 메모리에 구현된 실체이고 변수의 개념과 동일하다.

  2. 개발자가 클래스 내부에 별도 생성자를 만들지 않는다면 컴파일러는 인수가 없는 디폴트 생성자를 만들어 제공한다.

  3. 개발자가 클래스 내 생성자를 하나라도 만든다면, 컴파일러는 디폴트 생성자를 제공하지 않는다. 만약 디폴트 생성자가 필요하면, 개발자가 직접 만들어주거나 아니면 다음과 같이 선언한다. 

     ```c++
     Person() = default;
     ```

     > 개발자가 직접 제작한 디폴트 생성자와 컴파일러가 제공하는 디폴트 생성자는 기능상으로 약간의 차이가 존재한다.
     > 우리가 만든 디폴트 생성자 : 함수 호출 연산자의 존재 여부에 대한 구분 없이 모든 객체를 초기화시킨다.
     > 컴파일 제공 디폴트 생성자 : 함수 호출 연산자가 없다면 클래스나 구조체의 변수는 생성하지만 멤버 변수를 초기화시켜주지는 않는다.

  4. 클래스의 인스턴스는 생성자의 본문이 실행되기 이전에 메모리의 영역을 잡아 제공된다.
     따라서 생성자는 인스턴스화된 멤버 변수를 초기화하고, 객체가 생성되었다는 사실을 알려주는 목적으로 만들어진 함수이다.

  

- 클래스 생성자 예제 : 

  ```c++
  struct Account {
      char *account;
      char *name;
      int balance;
      
      // 디폴트 생성자
      Account() {
          // 변수를 초기화 한 후 포인터를 nullptr로 바꿈
          account = nullptr;
          name = nullptr;
          balance = 0;
      }
      
      // 사용자 정의 생성자
      Account(const char * id, const char * name, int bal) {
          this->account = new char[strlen(id) + 1];
          strcpy(this->account, id); // id의 문자열을 account로 복사
          
          this->name = new char[strlen(name) + 1];
          strcpy(this->name, name);
          this->balance = bal;
          std::cout << "Account 생성자가 호출되었습니다." << std::endl;
      }
      ... ...
  }
  ```

  > strcpy() : from <cstring>, strcpy(a,b) : b의 문자열을 a로 복사
  >
  > strlen() : from <cstring>, strlen(a) : a 문자열의 길이를 반환
  >
  > nullptr : 기존의 NULL을 대체한다. 참고자료 - https://psychoria.tistory.com/21

  만약 클래스 생성자가 없다면, 일일이 클래스의 멤버 변수에 대한 데이터를 입력시켜야 한다.

  

- **멤버 초기화 리스트**(Member Initializer Lists) : 

  멤버 변수를 초기화시키는 또다른 방법이며, 다른 이름으로 **직접 초기화**(Direct Initialization)이라고도 부른다.

  이는 다음과 같이 멤버 변수와 함께 ()나 {}를 사용하여 생성자의 인수를 입력하거나, 또 다른 생성자나 함수를 호출하여 멤버 변수를 초기화시키는 방법이다.

  ```c++
  struct Simple {
      int n;
      const int y;
      
      // 생성자 선언
      Simple (int);
      
      // 디폴트 생성자 선언. 직접 초기화를 통해 n은 멤버 변수 7로, y는 0으로 초기화한다.
      Simple () : n(7), y(0) {} 
  };
  
  // 앞에서 선언한 생성자를 정의한다.
  Simple::Simple(int x) : n{x}, y(x) {}
  ```

  멤버 초기화 리스트는 멤버 변수를 초기화시키는 것 이외에 다음과 같은 특징을 갖는다.

  1. 생성자의 본문을 호출하기 전에 초기화 작업을 수행
  2. 상수는 반드시 멤버 '초기화 리스트' 내에 명시되어야 설정 가능
  3. 필요하다면 클래스 내 존재하는 멤버 함수나 다른 생성자 호출 가능
  4. 상속받는 기본 클래스의 생성자나 함수를 호출 가능

  

  멤버 초기화 리스트는 앞서 언급한것처럼 같은 클래스 내에 존재하는 다른 생성자나 멤버 함수를 호출하는 기능을 제공한다.

  ```c++
  struct Account {
      char *account;
      char *name;
      int balance;
      
      Account(const char *id, const char *name, int bal) 
          : account(new char[strlen(id) + 1]), name(new char[strlen(name) + 1]), balance(bal) {
          strcpy(this->account, id);
          strcpy(this->name, name);
      }
      ... ...
  };
  ```

  위와 같이 생성자로 멤버를 초기화하면 다음과 같은 작업이 이루어진다.

  1. 생성자에 데이터를 전달하기 위해 변수를 복사하여 생성자의 인수를 만든다. 만약 인수가 포인터라면 변수의 주소를 복사한다.
  2. 클래스의 인스턴스를 생성하기 위해 스택 메모리 또는 힙 메모리 공간을 확보한다.
  3. 초기화 리스트 내 선언된 순서에 따라 객체 내 멤버 변수에 초기화하거나 함수나 생성자를 호출한다.
  4. 만약 생성자의 본문이 존재하면 운영체제는 생성자를 호출하고, 프로그램의 제어를 생성자의 본문으로 넘긴다.

  > c++ 개발자들은 초기화 리스트를 사용하는 방법을 선호하는데, 그 이유는 초기화 리스트를 사용하면 위와 같이 클래스 멤버의 변수를 초기화하는 작업을 생성자 본문 내부에서 수행하는 것이 아니라 운영체제가 객체를 생성하는 동시에 멤버를 초기화시킨다는 장점이 있기 때문이다.

  

- this 포인터

  this 포인터는 클래스나 구조체가 인스턴스화 되었을 때 객체 자신을 스스로 가리키는 기능이다. 이 뿐만 아니라 자신과 다른 객체와의 연산을 수행하거나 함수를 호출시 상호 객체를 구분해줄 필요가 있을때 등 다양한 목적으로 사용된다.

  this 포인터는 아래와 같은 특징을 갖는다.

  1. 별도 선언 없이 클래스나 구조체 내부에서 사용 가능하다.
  2. 자기 자신을 가리키는 포인터 형식으로 클래스 내부의 모든 변수나 함수 앞에 암시적으로 존재하는 것으로 인식한다.



## 5.2 접근 지정자(Access Specifier)

- 앞선 예제들은 구조체를 많이 사용하였다. 구조체의 특징은 다음과 같다.

  1. 구조체는 클래스와 달리 C언어로부터 물려받은 것이다.
  2. 구조체의 모든 멤버는 main()함수를 포함하여 다른 모든 함수로부터 접근이 가능하다.

  클래스나 구조체는 선언시 public이나 private와 같은 *접근 지정자* 를사용하여 외부 함수나 클래스가 자신의 멤버에 접근하는 것에 제한을 줄 수 있다.

  ```c++
  struct Example {
  public: // 접근 지정자
  	void add(int x) {
          n += x;
      }
  private: // 접근 지정자
      int n = 0;
  }
  ```



### 접근 지정자

- 접근 지정자(Access Specifier)는 외부의 함수 또는 클래스가 해당 클래스의 객체 내 멤버에 접근할 수 있는 허용레벨을 나타낸다.

- 접근 지정자의 종류는 다음과 같다.

  1. **public** : 클래스 외부 또는 클래스 내부에서 별도의 제한 없이 멤버 함수를 호출할 수 있고, 멤버 변수의 데이터에 대한 입/출력또한 가능하다.
  2. **private** : 접근 허용 수준이 가장 낮은 단계로, private으로 선언된 멤버의 접근은 오로지 해당 클래스나 해당 구조체 내의 멤버로 제한된다.
  3. **protected** : 다음과 같은 객체에 한하여 접근을 허용한다.
     - 클래스 내부 멤버
     - 클래스를 상속한 하위 클래스의 멤버

  따라서 접근 지정자의 허용 범위는 **public** > **protected** > **private** 이다.

  다음은 클래스 멤버에 대한 접근 지정자를 명시한 표 이다.

  | 타입   | 사용 가능한 접근 지정자    | 디폴트 값                                                    |
  | ------ | -------------------------- | ------------------------------------------------------------ |
  | enum   | 없음                       | 모두 public으로 인식하나, public을 지정자로 사용할 수는 없다. |
  | class  | public, protected, private | private                                                      |
  | struct | public, protected, private | public                                                       |



### 접근 지정자 예제

```c++
#include <cstdlib>
#include <cstdio>
#include <cstring>

struct Account {
    // 구조체의 접근 지정자 디폴트값은 public. 즉 다른 함수로부터 호출 가능
    Account(const char *id, const char *name, int bal) {
        strcpy(this->account, id);
        strcpy(this->name, name);
        this->balance = bal;
    }
    void Print() {
        printf("계좌 : %s, 소유자 : %s", account, name);
        printf(", 잔액 : %i\n", balance);
    }
    void Deposit(int money){
        balance += money;
    }
    void Withdraw(int money){
        balance-= money;
    }
    
// 만일 private을 사용하면 외부 함수 접근시 에러 발생
//private:
    char account[20];
    char name[20];
    int balance;
};

int main() {
    char id[] = "120-3450129099";
    char name[] = "홍길동";
    Account hong = Account(id, name, 60000);
    hong.Print();
    hong.Withdraw(10000);
    printf("출금 완료\n");
    printf("계좌 : %s, 소유자 : %s", hong.account, hong.name);
    printf(", 잔액 : %i\n", hong.balance);
    return 0;
}
```

실행 결과 : 

![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/c%2B%2B/5-2_1.png)



### 클래스와 구조체 멤버의 선언 순서

- 선언 순서와 관련된 별도 규칙은 존재하지 않으나 많은 회사가 개발의 편의를 위해 다음과 같은 방식으로 클래스 멤버를 선언할 것을 권고한다.

  1. 접근 지정자는 다음과 같이 public, protected와 private 순으로 나열한다.

  2. using 지정자와 typedef 지정자, 그리고 열거형 타입(enumeration)

     이들은 개발 시 가장 많이 사용하므로 접근 지정자를 선언한 위치 바로 다음에 선언한다.

  3. 내부 클래스

     다른 멤버보다 내부 클래스를 먼저 표시한다.

  4. 멤버 상수와 정적 멤버 변수

  5. 생성자(Constructors)

  6. 소멸자(Destructor)

  7. 멤버 함수나 정적 멤버 함수(static function)

  8. 연산자 오버로딩

  9. 일반 멤버 변수(멤버 상수와 정적 멤버 변수는 위에서 선언하므로 제외)

  ```c++
  class Classes {
    public:
      void setLength( double len );
      double getLength();
      double length;
      
    protected:
      void FuncProtected();
      int y;
      
    private:
      void FuncPrivate();
      int z;
  };
  ```

  > 연산자 오버로딩 참고자료 - https://blog.hexabrain.net/177



## 5.3 구조체와 클래스의 차이

- 구조체와 클래스의 차이는 다음과 같다.

  1. 클래스는 디폴트 접근 지정자가 private으로 선언되어 있다. 구조체는 public이 디폴트 접근 지정자이다.

  2. 템플릿 사용 시 template<class T>는 허용되지만 template<struct T>는 허용되지 않는다.

     이는 단순히 템플릿 인수를 선언하는 키워드로 struct를 사용할 수 없다는 의미이다.

     

- 구조체와 클래스를 분리하여 사용하는 목적은 다음과 같다.

  1. 구조체는 생성자와 소멸자가 없는 POD(plain old data structs)를 만들 때 사용한다
  2. 구조체는 주로 함수 위주가 아닌 변수 위주로 멤버를 구성할 때 우선적으로 사용한다.
  3. 구조체의 멤버로 생성자와 소멸자, 그리고 멤버를 관리하는 함수는 되도록 사용하지 않는다.
  4. 구조체는 클래스 타입과 달리 int와 long 타입의 기본 타입처럼 생성된 객체의 크기가 일정한 데이터를 다룰 때 주로 사용한다.



## 5.4 객체 초기화

- 완성된 프로그램을 유지 보수하거나 버그를 수정할때 클래스 내 멤버를 삭제하거나 수정하게 되면, 해당 멤버에 접근하여 데이터의 입출력을 수행하던 외부 함수나 프로그램들에 문제가 생긴다. 이러한 문제가 발생하지 않도록 수정되지 않아야 할 함수나 멤버 변수를 접근 지정자와 생성자로 관리한다.

- 만일 클래스 내 변수들을 private으로 선언한다면, 다음과 같은 장점이 생긴다.

  1. 외부에서 변수를 쉽게 수정하지 못한다.
  2. 오직 클래스 내부 생성자나 함수를 통해 변수 수정이 가능하다.
  3. 생성자를 사용하여 한번 생성된 클래스 내 변수는 별도 함수를 제공하지 않는 한 수정을 비롯하여 읽을 수 있는 방법 조차도 제공되지 않는다.

- 정적 멤버 변수

  클래스 내 정적 멤버 변수 또는 정적 멤버 상수는 다음과 같은 특징을 갖는다.

  1. 정적(static)으로 선언된 변수와 상수는 클래스 외부에서 초기화해야 한다.
  2. 정적(static)으로 선언된 상수는 객체 생성 이전에 이미 존재하는 값이기 때문에 생성자를 사용하여 정적 상수를초기화 시킬 수 없다.

  ```c++
  #include <iostream>
  
  class Something {
  private:
      static int s_value; // 정적 멤버 변수
      static const int c_value; // 정적 멤버 상수
  public:
      static int getValue() { return s_value; } // 정적 함수 호출 시 일반 멤버 변수 사용 불가.
      static const int getConst() { return c_value; } 
  };
   
  int Something::s_value = 1; // 정적 변수를 초기화한다.
  const int Something::c_value = 10; // 정적 상수를 초기화한다.
  
  int main() {
      // 클래스 변수 선언이 없어도 정적 변수는 언제든 사용 가능
      std::cout << "Something::getValue() = " << Something::getValue() << std::endl;
      std::cout << "Something::getConst() = " << Something::getConst() << std::endl;
  }
  ```

  실행 결과 : 

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/c%2B%2B/5-2_2.png)



- 정적?

  아래 그림은 c++ 프로그램의 메모리 영역이다. 정적 변수는 Data의 'read/write segment' 영역으로 이동하고, 상수는 'read-only segment' 영역으로 이동하여 보관한다.

  정적이 아닌 다른 변수들은 모두 stack이나 heap 영역에 보관되어 생성과 소멸이 반복된다.

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/c%2B%2B/5-2_3.png)

  예시 코드 : 

  ```c++
  #include <iostream>
  
  class MyClass {
    public:
      MyClass(int i, int j) {
         x = i;
         y = j;
      }
  
      void ShowXY() {
        std::cout << "The field values are " << x << " & " << y << std::endl; 
      }
  
    private:
      static int x;
      int y;
  };
  
  int MyClass::x = 0;
  
  int main() {
          MyClass ms1(10, 20);
          MyClass ms2(30, 50);
          ms1.ShowXY();
          ms2.ShowXY();
  }
  ```

  실행 결과 : 

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/c%2B%2B/5-2_4.png)



- 정적 멤버 사용의 목적 : 

  1. 클래스의 정적 멤버 함수는 이미 존재하기 때문에 객체가 생성되지 않더라도 호출이 가능하다.

     > 대표적인 예로 new가 있다. new 연산자는 클래스가 객체화되기 전에 요구하는 메모리 공간을 잡기 위해 정적 연산자로 ㄹ만든다.

  2. 클래스의 정적 멤버 변수는 전역 변수와 유사하다. 차이가 있다면 전역 변수는 클래스를 포함하여 모든 프로그램이 공유하지만, 정적 멤버 변수는 접근 지정자의 설정에 따라 특정 클래스 내에서 공유한다.

