import React, { useState } from "react";
import { Button, Input, Alert, CircularProgress } from "@mui/material";
import axios from "axios";
import { BASE_URL } from "../constants";

export default function CSVImporter({ onPlayersLoaded }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [players, setPlayers] = useState({});
  const [alert, setAlert] = useState({
    message: "",
    severity: "",
    show: false,
  });
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleFileUpload = async () => {
    if (!selectedFile) {
      setAlert({
        message: "Please select a CSV file first.",
        severity: "error",
        show: true,
      });
      return;
    }
    const formData = new FormData();
    formData.append("file", selectedFile);

    setLoading(true);

    try {
      const response = await axios.post(
        `${BASE_URL}/match/upload_csv`,
        formData
      );

      const result = response.data;

      if (result) {
        setPlayers(result);
        onPlayersLoaded(result); // Pass the players data back to the parent
        setAlert({
          message: "Successfully imported players from the CSV file.",
          severity: "success",
          show: true,
        });
      } else {
        setAlert({
          message:
            "Failed to retrieve player data. Please check the CSV format.",
          severity: "error",
          show: true,
        });
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      setAlert({
        message: "Error uploading file. Please try again.",
        severity: "error",
        show: true,
      });
    } finally {
      setLoading(false);
    }
  };

  const handleAlertClose = () => {
    setAlert({
      message: "",
      severity: "",
      show: false,
    });
  };

  return (
    <div className="card flex justify-content-center">
      {alert.show && (
        <Alert
          severity={alert.severity}
          sx={{ marginBottom: 2, width: "100%" }}
          onClose={handleAlertClose}
        >
          {alert.message}
        </Alert>
      )}
      <Input
        type="file"
        inputProps={{ accept: ".csv" }}
        onChange={handleFileChange}
        sx={{ marginBottom: 2, backgroundColor: "transparent" }}
      />
      <Button
        variant="contained"
        color="success"
        onClick={handleFileUpload}
        disabled={!selectedFile || loading}
        endIcon={loading ? <CircularProgress size={24} /> : null}
      >
        {loading ? "Importing..." : "Import from CSV"}
      </Button>
    </div>
  );
}
