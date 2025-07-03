#!/bin/bash

# Helper script for common operations with the Aegis RAG system

# Colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to display help message
show_help() {
    echo -e "${BLUE}Aegis RAG System Helper${NC}"
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Available commands:"
    echo "  start         - Start all containers"
    echo "  stop          - Stop all containers"
    echo "  restart       - Restart all containers"
    echo "  status        - Show container status"
    echo "  logs [service]- Show logs for all or specific service"
    echo "  ingest        - Restart ingestor to process documents in data/raw"
    echo "  rebuild       - Rebuild and restart all containers"
    echo "  clean         - Remove all containers and volumes (data will be lost)"
    echo "  help          - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh start"
    echo "  ./run.sh logs api"
}

# Check if Docker is running
check_docker() {
    if ! sudo docker info > /dev/null 2>&1; then
        echo -e "${RED}Error: Docker is not running. Please start Docker first.${NC}"
        exit 1
    fi
}

# Main command switch
case "$1" in
    start)
        check_docker
        echo -e "${GREEN}Starting Aegis RAG system...${NC}"
        docker compose up -d
        echo -e "${GREEN}System started. UI available at http://localhost:8920${NC}"
        ;;
    stop)
        check_docker
        echo -e "${YELLOW}Stopping Aegis RAG system...${NC}"
        docker compose down
        echo -e "${GREEN}System stopped.${NC}"
        ;;
    restart)
        check_docker
        echo -e "${YELLOW}Restarting Aegis RAG system...${NC}"
        docker compose down
        docker compose up -d
        echo -e "${GREEN}System restarted. UI available at http://localhost:8920${NC}"
        ;;
    status)
        check_docker
        echo -e "${BLUE}Container status:${NC}"
        docker compose ps
        ;;
    logs)
        check_docker
        if [ -z "$2" ]; then
            echo -e "${BLUE}Showing logs for all services:${NC}"
            docker compose logs
        else
            echo -e "${BLUE}Showing logs for $2:${NC}"
            docker compose logs "$2"
        fi
        ;;
    ingest)
        check_docker
        echo -e "${YELLOW}Restarting ingestor to process documents...${NC}"
        docker compose restart ingestor
        echo -e "${GREEN}Ingestor restarted. Check logs with: ./run.sh logs ingestor${NC}"
        ;;
    rebuild)
        check_docker
        echo -e "${YELLOW}Rebuilding and restarting all containers...${NC}"
        docker compose down
        docker compose up -d --build
        echo -e "${GREEN}System rebuilt and restarted. UI available at http://localhost:8920${NC}"
        ;;
    clean)
        check_docker
        echo -e "${RED}WARNING: This will remove all containers and volumes. All data will be lost.${NC}"
        read -p "Are you sure you want to continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Removing all containers and volumes...${NC}"
            docker compose down -v
            echo -e "${GREEN}Cleanup complete.${NC}"
        else
            echo -e "${BLUE}Operation cancelled.${NC}"
        fi
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac 