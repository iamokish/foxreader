/* Tiny owned-backend fixture: no Python, models, network downloads or user data. */
#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#define close_socket closesocket
#define pause_ms Sleep
#else
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#define close_socket close
#define pause_ms(ms) usleep((ms) * 1000)
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    struct sockaddr_in address;
    const char *port = getenv("FOX_TEST_PORT"), *token = getenv("FOX_READER_SHUTDOWN_TOKEN");
    const char *delay = getenv("FOX_TEST_DELAY");
    const char *html = "<!doctype html><html><head><meta name='color-scheme' content='dark light'></head>"
        "<body style='font:24px system-ui;padding:60px'><h1>Fox Reader</h1>"
        "<p>Native viewer test: backend ready.</p>"
        "<script>fetch('/ui-ready')</script></body></html>";
    FILE *file;
    int yes = 1, stop = 0;
    unsigned int ready_pages = 0;
#ifdef _WIN32
    SOCKET listener, peer;
    WSADATA data;
    WSAStartup(MAKEWORD(2, 2), &data);
#else
    int listener, peer;
#endif
    if (!port || !token) return 2;
    file = fopen("started", "wb");
    if (file) { fputs("started", file); fclose(file); }
    pause_ms(delay ? atoi(delay) : 200);
    listener = socket(AF_INET, SOCK_STREAM, 0);
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, (const char *)&yes, sizeof yes);
    memset(&address, 0, sizeof address);
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons((unsigned short)atoi(port));
    if (bind(listener, (struct sockaddr *)&address, sizeof address) || listen(listener, 8)) return 3;
    fputs("native backend fixture ready\n", stdout);
    fflush(stdout);
    while (!stop) {
        char request[8192], reply[4096];
        const char *body = "{\"app\":\"fox-reader\"}";
        int count = 0, code = 200;
#ifdef _WIN32
        DWORD timeout = 1000;
#else
        struct timeval timeout = {1, 0};
#endif
        peer = accept(listener, NULL, NULL);
        setsockopt(peer, SOL_SOCKET, SO_RCVTIMEO, (const char *)&timeout, sizeof timeout);
        request[0] = 0;
        while (count < (int)sizeof request - 1 && !strstr(request, "\r\n\r\n")) {
            int received = recv(peer, request + count, sizeof request - 1 - count, 0);
            if (received <= 0) break;
            count += received;
            request[count] = 0;
        }
        if (count <= 0) { close_socket(peer); continue; }
        request[count] = 0;
        if (strncmp(request, "POST /api/shutdown ", 19) == 0) {
            if (strstr(request, token)) stop = 1;
            else code = 403;
        } else if (strncmp(request, "GET /api/health ", 16) != 0) {
            body = html;
            if (strncmp(request, "GET / ", 6) == 0) {
                file = fopen("ui-requested", "wb");
                if (file) { fputs("loaded", file); fclose(file); }
            }
            if (strncmp(request, "GET /ui-ready ", 14) == 0) {
                /* Separate evidence for the owner's and attached view's page;
                 * tests must not mistake the first view for a ready second one. */
                char marker[32];
                snprintf(marker, sizeof marker, "ui-ready-%u", ++ready_pages);
                file = fopen(marker, "wb");
                if (file) { fputs("javascript ready", file); fclose(file); }
            }
        }
        count = snprintf(reply, sizeof reply,
            "HTTP/1.1 %d OK\r\nContent-Type: %s\r\nContent-Length: %u\r\n"
            "Cache-Control: no-store\r\nConnection: close\r\n\r\n%s",
            code, body == html ? "text/html" : "application/json", (unsigned)strlen(body), body);
        send(peer, reply, count, 0);
        close_socket(peer);
    }
    close_socket(listener);
    file = fopen("shutdown-ok", "wb");
    if (file) { fputs("token accepted", file); fclose(file); }
    return 0;
}
