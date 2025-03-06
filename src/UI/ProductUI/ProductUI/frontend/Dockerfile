# Use a Node.js image as the base image
FROM node:20

# Set the working directory inside the container
WORKDIR /frontend

# Copy package.json and package-lock.json to install dependencies
COPY package.json package-lock.json ./

# Install the dependencies
RUN npm install

# Copy the rest of the frontend files
COPY . .

# Expose the port on which the frontend app runs
EXPOSE 3000

# Command to run the frontend app
CMD ["npm", "run", "dev"]
