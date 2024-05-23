#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <windows.h>

#define N 3
using namespace std;

/*
0 - empty
1 - ship
2 - hit
3 - miss
*/

void place_ship(int ship_size, int x, int y, int dir, int gameboard[N][N])
{
    for (int i = 0; i < ship_size; i++)
    {
        switch (dir)
        {
        case 0:
            gameboard[x + i][y] = 1;
            break;
        case 1:
            gameboard[x][y + i] = 1;
            break;
        case 2:
            gameboard[x - i][y] = 1;
            break;
        case 3:
            gameboard[x][y - i] = 1;
            break;
        }
    }
}

void rand_ship(int ship_size, int gameboard[N][N])
{
    int x = rand() % N;
    int y = rand() % N;
    int dir = rand() % 4;
    switch (dir)
    {
    case 0:
        if (x + ship_size - 1 < N)
        {
            break;
        }
        else
        {
            dir = 2;
            break;
        }
    case 1:
        if (y + ship_size - 1 < N)
        {
            break;
        }
        else
        {
            dir = 3;
            break;
        }
    case 2:
        if (x - ship_size + 1 >= 0)
        {
            break;
        }
        else
        {
            dir = 0;
            break;
        }
    case 3:
        if (y - ship_size + 1 >= 0)
        {
            break;
        }
        else
        {
            dir = 1;
            break;
        }
    }
    place_ship(ship_size, x, y, dir, gameboard);
}
void player_ship(int ship_size, int gameboard[N][N])
{
    int x = 0, y = 0, dir = 0;
    do
    {
        cout << "Provide ship front x (0-9)";
        cin >> x;
        cout << "Provide ship front y (0-9)";
        cin >> y;
        cout << "Provide ship direction (0 -down, 1 - right, 2 - up, 3 - left)";
        cin >> dir;
    } while (x < 0 || x >= N || y < 0 || y >= N || dir < 0 || dir > 3 || (dir == 0 && x + ship_size - 1 >= N) || (dir == 1 && y + ship_size - 1 >= N) || (dir == 2 && x - ship_size + 1 < 0) || (dir == 3 && y - ship_size + 1 < 0));
    place_ship(ship_size, x, y, dir, gameboard);
}

void print_full_board(int gameboard[N][N])
{
    for (int i = 0; i < N; i++)
    {
        cout << "\t" << i;
    }
    cout << "\n";
    for (int i = 0; i < N; i++)
    {
        cout << i << "\t";
        for (int j = 0; j < N; j++)
        {
            switch (gameboard[i][j])
            {
            case 0:
                cout << "~\t";
                break;
            case 1:
                cout << "#\t";
                break;
            case 2:
                cout << "X\t";
                break;
            case 3:
                cout << "O\t";
                break;
            }
        }
        cout << "\n";
    }
    cout << endl;
}

void print_enemy_board(int gameboard[N][N])
{
    cout << "\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9\n";
    for (int i = 0; i < N; i++)
    {
        cout << i << "\t";
        for (int j = 0; j < N; j++)
        {
            switch (gameboard[i][j])
            {
            case 0:
                cout << "~\t";
                break;
            case 1:
                cout << "~\t"; // todo remove enemy ships from sight
                break;
            case 2:
                cout << "X\t";
                break;
            case 3:
                cout << "O\t";
                break;
            }
        }
        cout << "\n";
    }
    cout << endl;
}

bool enemy_random_shot(int gameboard[N][N])
{
    int x = 0, y = 0;
    do
    {
        x = rand() % N;
        y = rand() % N;
    } while (gameboard[x][y] == 2 || gameboard[x][y] == 3);
    if (gameboard[x][y] == 1)
    {
        gameboard[x][y] = 2;
        return true;
    }
    else
    {
        gameboard[x][y] = 3;
        return false;
    }
}

bool player_shot(int gameboard[N][N])
{
    int x, y;
    do
    {
        cout << "Give x coordinate:";
        cin >> x;
        cout << "Give y coordinate:";
        cin >> y;
    } while (x < 0 || x >= N || y < 0 || y >= N);
    if (gameboard[x][y] == 1)
    {
        gameboard[x][y] = 2;
        cout << "HIT!\n";
        return true;
    }
    else
    {
        gameboard[x][y] = 3;
        cout << "miss... \n";
        return false;
    }
}

bool check_win(int gameboard[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (gameboard[i][j] == 1)
            {
                return false;
            }
        }
    }
    return true;
}

void print_boards(int player_gameboard[N][N], int computer_gameboard[N][N])
{
    cout << "Player board:\n";
    print_full_board(player_gameboard);
    cout << "Enemy board:\n";
    print_enemy_board(computer_gameboard);
}

void chk_win(int player_gameboard[N][N], int computer_gameboard[N][N])
{
    if (check_win(computer_gameboard))
    {
        cout << "You have won!\n";
        print_boards(player_gameboard, computer_gameboard);
        exit(0);
    }
    if (check_win(player_gameboard))
    {
        cout << "You have lost...\n";
        print_boards(player_gameboard, computer_gameboard);
        exit(0);
    }
}

int main()
{
    int player_gameboard[N][N];
    int computer_gameboard[N][N];
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            player_gameboard[i][j] = 0;
            computer_gameboard[i][j] = 0;
        }
    }
    uint64_t turn_counter = 1;
    srand(time(NULL));
    vector<int> ship_sizes = {2, 2};
    for (uint16_t i = 0; i < ship_sizes.size(); i++)
    {
        cout << "Placing ship no." << i + 1 << " of " << ship_sizes.size() << "\n Size: " << ship_sizes[i] << "\n";
        player_ship(ship_sizes[i], player_gameboard);
        cout << "Player ship placed\n";
        print_full_board(player_gameboard);
        rand_ship(ship_sizes[i], computer_gameboard);
    }

    cout << "GAME STARTED\n";
    print_boards(player_gameboard, computer_gameboard);

    bool enemy_success, player_success;
    while (turn_counter != 0)
    {
        enemy_success = true;
        player_success = true;
        cout << "Turn " << turn_counter << "\n";
        while (enemy_success)
        {
            Sleep(3000);
            cout << "Enemy shooting!\n";
            enemy_success = enemy_random_shot(player_gameboard);
            chk_win(player_gameboard, computer_gameboard);
            if (enemy_success)
            {
                cout << "We are hit...\n";
            }
            else
            {
                cout << "That missed us!\n";
            }
            print_boards(player_gameboard, computer_gameboard);
        }
        while (player_success)
        {
            cout << "Your shot!\n";
            player_success = player_shot(computer_gameboard);
            chk_win(player_gameboard, computer_gameboard);
            print_boards(player_gameboard, computer_gameboard);
        }
        turn_counter++;
    }
}