/* Exercise the launcher's real config reader without opening any window. */
#define main fox_launcher_main
#include "../packaging/launcher/launcher.c"
#undef main

int main(void)
{
    char host[MAX_HOST];
    int port;
    read_config(PL("."), host, sizeof host, &port);
    printf("%s\n%d\n", host, port);
    return 0;
}
