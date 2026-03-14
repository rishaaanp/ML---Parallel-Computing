#include <iostream>
#include <pthread.h>

using namespace std;

int n;

// Function executed by the thread
void* printNumbers(void* arg)
{
    cout << "First " << n << " natural numbers are:\n";

    for(int i = 1; i <= n; i++)
    {
        cout << i << " ";
    }

    cout << endl;

    pthread_exit(NULL);
}

int main()
{
    pthread_t thread;

    cout << "Enter value of n: ";
    cin >> n;

    // Create thread
    pthread_create(&thread, NULL, printNumbers, NULL);

    // Wait for thread to finish
    pthread_join(thread, NULL);

    return 0;
}